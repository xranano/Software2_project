"""Project task — lane follow + object detection on real hardware."""
import importlib.util
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

odet_path = os.path.abspath(os.path.join(script_dir, '..', 'object_detection', 'real_server.py'))
_spec = importlib.util.spec_from_file_location('object_detection_real_server', odet_path)
real_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(real_server)

real_server.HTML_TEMPLATE = real_server.HTML_TEMPLATE.replace('Object Detection', 'Project')

if __name__ == '__main__':
    sys.exit(real_server.main())
