import os
import json
import datetime
from typing import List, Dict, Any, Tuple, Optional
from rich.console import Console


class AgentLogger:
    def __init__(self, model: str, task: str, lab: str):
        self.model = model
        self.task = task
        self.console = Console()
        self.log_data = []
        self.log_dir = f"../results/{lab}/langgraph/{model}/{task}"
        
        # Get program start time as timestamp
        self.start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Use model_task_timestamp format as log filename
        self.log_path = f"{self.log_dir}/{model}_{task}_{self.start_timestamp}.json"
        
        # Check if file with same name already exists, add run number if needed
        self._ensure_unique_log_path()
        
    def _ensure_unique_log_path(self):
        """Ensure log path is unique, add run number suffix if file with same name exists"""
        original_path = self.log_path
        counter = 1
        
        while os.path.exists(self.log_path):
            # If same timestamp exists, add run number suffix
            self.log_path = f"{original_path.rstrip('.json')}_run{counter}.json"
            counter += 1
    
    def log_agent_output(self, content: Any, agent_name: str, role: str = "assistant"):
        """Record agent output"""
        if isinstance(content, (list, dict)):
            # Convert to string if complex data structure
            content_str = json.dumps(content, indent=2)
        else:
            content_str = str(content)
            
        # Print to console
        self.console.print(f"[bold cyan]--- {agent_name} ({role}) ---[/bold cyan]")
        self.console.print(content_str)
        
        # Append to log
        self.append_log(content, agent_name, role)
    
    def log_message(self, message: Tuple[str, str], agent_name: str):
        """Record message conversation"""
        role, content = message
        self.append_log(content, agent_name, role)
    
    def log_messages(self, messages: List[Tuple[str, str]], agent_name: str):
        """Record multiple messages"""
        for message in messages:
            self.log_message(message, agent_name)
    
    def append_log(self, content: Any, agent_name: str, role: str):
        """Add entry to log"""
        log_entry = {
            "content": content,
            "role": role,
            "name": agent_name,
        }
        self.log_data.append(log_entry)
        
        # Save log in real-time
        self.save_logs()
    
    def log_result(self, result: Any):
        """Record final result"""
        self.append_log(result, agent_name="result", role="system")
        
    def save_logs(self):
        """Save logs to file"""
        with open(self.log_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)
            
    def get_formatted_logs(self):
        """Get formatted log data"""
        return self.log_data