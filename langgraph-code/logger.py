import os
import json
import datetime
from typing import List, Dict, Any, Tuple, Optional
from rich.console import Console


class AgentLogger:
    def __init__(self, model: str, task: str, lab: str):
        """
        Initialize the agent logger.
        
        Args:
            model: The model name being used
            task: The task being performed
        """
        self.model = model
        self.task = task
        self.console = Console()
        self.log_data = []
        self.log_dir = f"../results/{lab}/react/{model}/{task}"
        
        # Get program start time as timestamp
        self.start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Use model_task_timestamp format as log filename
        self.log_path = f"{self.log_dir}/{model}_{task}_{self.start_timestamp}.json"
        
        # Check if a file with the same name already exists, if so add a run number
        self._ensure_unique_log_path()
        
    def _ensure_unique_log_path(self):
        """Ensure log path is unique, if a file with the same name exists, add a run number suffix"""
        original_path = self.log_path
        counter = 1
        
        while os.path.exists(self.log_path):
            # If the same timestamp already exists, add a run number suffix
            self.log_path = f"{original_path.rstrip('.json')}_run{counter}.json"
            counter += 1
    
    def log_agent_output(self, content: Any, agent_name: str, role: str = "assistant"):
        """
        Log agent output.
        
        Args:
            content: The content to log
            agent_name: The name of the agent
            role: The role of the agent (default: "assistant")
        """
        if isinstance(content, (list, dict)):
            # If it's a complex data structure, convert to string
            content_str = json.dumps(content, indent=2)
        else:
            content_str = str(content)
            
        # Print to console
        self.console.print(f"[bold cyan]--- {agent_name} ({role}) ---[/bold cyan]")
        self.console.print(content_str)
        
        # Append to log
        self.append_log(content, agent_name, role)
    
    def log_message(self, message: Tuple[str, str], agent_name: str):
        """
        Log a message conversation.
        
        Args:
            message: A tuple of (role, content)
            agent_name: The name of the agent
        """
        role, content = message
        self.append_log(content, agent_name, role)
    
    def log_messages(self, messages: List[Tuple[str, str]], agent_name: str):
        """
        Log multiple messages.
        
        Args:
            messages: A list of message tuples
            agent_name: The name of the agent
        """
        for message in messages:
            self.log_message(message, agent_name)
    
    def append_log(self, content: Any, agent_name: str, role: str):
        """
        Add an entry to the log.
        
        Args:
            content: The content to log
            agent_name: The name of the agent
            role: The role of the agent
        """
        log_entry = {
            "content": content,
            "role": role,
            "name": agent_name,
        }
        self.log_data.append(log_entry)
        
        # Save logs in real-time
        self.save_logs()
    
    def log_result(self, result: Any):
        """
        Log the final result.
        
        Args:
            result: The result to log
        """
        self.append_log(result, agent_name="result", role="system")
        
    def save_logs(self):
        """Save logs to file"""
        with open(self.log_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)
            
    def get_formatted_logs(self):
        """Get formatted log data"""
        return self.log_data