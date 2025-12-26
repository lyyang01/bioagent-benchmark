from logger import AgentLogger

# Global logger instance
_logger_instance = None

def init_logger(model, task, lab):
    """Initialize logger instance"""
    global _logger_instance
    _logger_instance = AgentLogger(model=model, task=task, lab=lab)
    return _logger_instance

def get_logger():
    """Get current logger instance"""
    return _logger_instance