import types
import json
import os
import psutil

class MemoryManager:
    def __init__(self):
        self.storage: Dict[str, Any] = {}
    
    def update(self, namespace):
        for key, value in namespace.items():
            if not key.startswith("__") and not isinstance(value, (types.FunctionType, type)):
                try:
                    self.storage[key] = value
                except:
                    pass
    
    def get_namespace(self):
        return self.storage.copy()

def convert_to_serializable(obj):
    """Convert object to a JSON serializable format"""
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        # If object has a to_dict method, call it
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        # If object has a __dict__ attribute, use it
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif isinstance(obj, dict):
        # Recursively process dictionaries
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively process lists
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively process tuples
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Return basic types directly
        return obj
    else:
        # Convert other types to string
        return str(obj)
    
def serialize_to_json(data, indent=2):
    """Serialize data to JSON string"""
    return json.dumps(data, default=convert_to_serializable) #json.dumps(data, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o), indent=2)



def kill_child_processes(parent_pid=None):

    if parent_pid is None:
        parent_pid = os.getpid()
    
    parent = psutil.Process(parent_pid)
    children = parent.children(recursive=True)  
    
    for child in children:
        try:
            child.terminate()
        except:
            pass
    
    gone, still_alive = psutil.wait_procs(children, timeout=3)
    
    for process in still_alive:
        try:
            process.kill()
        except:
            pass