import json
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from flask import Flask, request
import socketio

from robosuite.devices.device import Device


class IPhoneDevice(Device):
    """
    iPhone device for teleoperating the Panda-Omron base using iPhone orientation.
    Rotation-only mapping: delta roll -> lateral, delta pitch -> forward/backward, delta yaw -> rotation.
    """
    
    def __init__(self, env, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.0, port: int = 5557):
        super().__init__(env)

        self._display_controls()
        
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self._port = port
        self._latest_transform = {}
        self._neutral_quat = None
        self._prev_action = np.zeros(3)
        self._smoothing_factor = 0.2
        
        # socketio server setup
        self._sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
        self._app = Flask(__name__)
        self._app.wsgi_app = socketio.WSGIApp(self._sio, self._app.wsgi_app)
        
        # control state
        self._reset = False
        self._grasp = False
        self._base_mode = True

        self._connected = False

        @self._sio.event
        def connect(sid, environ):
            self._connected = True
            print(f"iphone connected: {sid}")
            print(f"user agent: {environ.get('HTTP_USER_AGENT', 'Unknown')}")

        @self._sio.event
        def disconnect(sid):
            self._connected = False
            print(f"iphone disconnected: {sid}")

        @self._sio.event
        def update(sid, data):
            try:
                print(f"received data: {type(data)}")
                
                if isinstance(data, str):
                    data_dict = json.loads(data)
                else:
                    data_dict = data
                
                self._latest_transform = data_dict
                print(f"data keys: {list(data_dict.keys())}")
                # for key, value in data_dict.items():
                #     print(f"{key} : {value}")

                # process transform matrix into orientation quaternion
                if 'transformMatrix' in data_dict:
                    transform_matrix = np.array(data_dict['transformMatrix'])
                    rotation_matrix = transform_matrix[:3, :3]
                    
                    rotation = R.from_matrix(rotation_matrix)
                    quat = rotation.as_quat() # x, y, z, w

                    data_dict['orientation'] = {
                        'x': quat[0],
                        'y': quat[1],
                        'z': quat[2],
                        'w': quat[3],
                    }

                # set neutral orientation first time we get data
                if self._neutral_quat is None and 'orientation' in data_dict:
                    self._set_neutral_orientation(data_dict['orientation'])
                    
            except Exception as e:
                print(f"iphone data update failed: {e}")
                import traceback
                traceback.print_exc()

        # catch all events
        @self._sio.event
        def message(sid, data):
            print(f"generic message: {type(data)} - {str(data)[:100]}...")

    @staticmethod
    def _display_controls():
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("iPhone Motion", "Commands")
        print_command("tilt forward/backward", "move forward/backward")
        print_command("tilt left/right", "move left/right")
        print_command("twist counter-clockwise/clockwise", "rotate counter-clockwise/clockwise")
        print_command("return to upright", "stop all motion")
        print("")

    def _set_neutral_orientation(self, orientation_dict):
        self._neutral_quat = np.array([
            orientation_dict['x'],
            orientation_dict['y'], 
            orientation_dict['z'],
            orientation_dict['w']
        ])
        print("neutral orientation set")
        print(f"   neutral quaternion: {self._neutral_quat}")

    def _run_server(self):
        print(f"starting iphone server on port {self._port}...")
        self._app.run(host="0.0.0.0", port=self._port, threaded=True, debug=False)

    def start_control(self):
        """start control - start the server and reset internal state."""
        self._reset_internal_state()
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()
        print(f"iphone server running at http://0.0.0.0:{self._port}")
        # needs to be connected here

    def get_controller_state(self):
        control_dict = self.get_control()
        
        if control_dict is None:
            if self._connected and not self._latest_transform:
                print("iphone connected, no data received yet")
            return {
                "dpos": np.zeros(3),
                "rotation": np.zeros(3),  # absolute rotation
                "raw_drotation": np.zeros(3),  # raw delta rotation
                "grasp": self._grasp,
                "reset": self._reset,
                "base_mode": self._base_mode,
            }
        
        # map base controls to 3D position and rotation commands
        # lateral->x, forward->y, rotation->z rotation
        dpos = np.array([control_dict['base_lateral'], control_dict['base_forward'], 0.0])
        
        # rotation: use raw delta rotation
        raw_drotation = np.array([0.0, 0.0, control_dict['base_rotation']])
        
        # absolute rotation
        rotation = np.zeros(3)
        
        return {
            "dpos": dpos,
            "rotation": rotation,
            "raw_drotation": raw_drotation,
            "grasp": self._grasp,
            "reset": self._reset,
            "base_mode": self._base_mode,
        }

    def get_control(self):
        if not self._latest_transform or 'orientation' not in self._latest_transform:
            return None
            
        if self._neutral_quat is None:
            print("no neutral orientation set yet")
            return None
            
        # current orientation
        orient_data = self._latest_transform['orientation']
        current_quat = np.array([
            orient_data['x'],
            orient_data['y'],
            orient_data['z'],
            orient_data['w']
        ])
        
        # relative rotation from neutral
        neutral_rot = R.from_quat(self._neutral_quat)
        current_rot = R.from_quat(current_quat)
        relative_rot = neutral_rot.inv() * current_rot
        
        # to euler angles (roll, pitch, yaw)
        euler_angles = relative_rot.as_euler('xyz', degrees=False)
        roll, pitch, yaw = euler_angles
        
        print(f"raw angles - roll: {roll:.3f}, pitch: {pitch:.3f}, yaw: {yaw:.3f}")
        
        # map to base controls, negative for control
        lateral = -pitch * self.pos_sensitivity
        forward = yaw * self.pos_sensitivity
        rotation = roll * self.rot_sensitivity
        
        # prevent drift when phone is near neutral
        deadzone = 0.05
        if abs(lateral) < deadzone: lateral = 0
        if abs(forward) < deadzone: forward = 0  
        if abs(rotation) < deadzone: rotation = 0
        
        # apply low-pass filtering for smoothing
        current_action = np.array([lateral, forward, rotation])
        smoothed_action = (1 - self._smoothing_factor) * self._prev_action + self._smoothing_factor * current_action
        self._prev_action = smoothed_action
        
        lateral, forward, rotation = smoothed_action
        
        # clamp values
        lateral = np.clip(lateral, -1.0, 1.0)
        forward = np.clip(forward, -1.0, 1.0) 
        rotation = np.clip(rotation, -1.0, 1.0)
        
        print(f"base control - lateral: {lateral:.3f}, forward: {forward:.3f}, rotation: {rotation:.3f}")
        
        # action dict for panda base control
        return {
            'base_lateral': lateral,
            'base_forward': forward, 
            'base_rotation': rotation
        }

    def _postprocess_device_outputs(self, dpos, drotation):
        return dpos, drotation

    def start(self):
        self.start_control()

    def stop(self):
        print("iphone stopped")


if __name__ == "__main__":
    import time
    
    class MockEnv:
        def __init__(self):
            class MockRobot:
                def __init__(self):
                    self.arms = ['right']
            self.robots = [MockRobot()]
    
    mock_env = MockEnv()
    device = IPhoneDevice(env=mock_env)
    device.start_control()
    
    print("\nwaiting for iphone connection...")
    print("   http://172.16.101.39:5557")
    
    try:
        while True:
            action = device.get_control()
            if action:
                print(f"control active: {action}")
            elif device._connected:
                print("iphone connected, waiting for orientation data")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        device.stop()