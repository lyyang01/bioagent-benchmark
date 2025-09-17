import io
import traceback
import warnings
from contextlib import redirect_stdout
import pandas as pd
from typing import Dict, Any, Optional
from rpy2.rinterface_lib import callbacks

class REnvironmentManager:
    """
    Manage the R execution environment with persistence support
    """
    def __init__(self):
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            # Set silent callback function to suppress interactive prompts
            def quiet_console(prompt):
                return None
            
            # Replace default console read function to avoid "Selection:" prompts
            callbacks.consoleread = quiet_console
            
            # Add callback to handle yes/no prompts
            def quiet_yes_no(question):
                # Automatically answer "no" for all yes/no questions
                return False
            
            # Replace console output and file selection callbacks
            callbacks.consolewrite = lambda x: None  # Suppress console output
            callbacks.choosefile = lambda x: ""      # Avoid file selection dialogs
            
            # Add callback to automatically handle yes/no questions
            if hasattr(callbacks, 'yesnocancel'):
                callbacks.yesnocancel = quiet_yes_no

            # Activate automatic conversion between pandas DataFrame and R data frame
            pandas2ri.activate()
            
            # Set R options to disable interactive features
            robjects.r('''
            options(menu.graphics=FALSE)
            options(repos=c(CRAN="https://cloud.r-project.org"))
            options(warn=-1)
            options(interactive=FALSE)
            options(askYesNo=function(...) FALSE)  # Automatically answer "no" to all prompts
            options(readLines=function(...) "")    # Avoid readLines prompts
            options(Bioconductor=list(askYesNo=FALSE))  # Bioconductor settings
            options(ExperimentHub.options=list(ask=FALSE, cache=tempdir())) # ExperimentHub settings
            options(BiocManager.check_repositories=FALSE)  # Disable repository check
            ''')
            
            # Use R's global environment
            self.r_env = robjects.globalenv
            self.robjects = robjects
            self.pandas2ri = pandas2ri
            self.localconverter = localconverter
            self.is_initialized = True
        except ImportError:
            print("Warning: rpy2 package is not installed, R environment manager will not work properly")
            self.is_initialized = False
    
    def get_environment(self):
        """
        Get the global R environment
        """
        if not self.is_initialized:
            raise RuntimeError("R environment manager not properly initialized")
        return self.r_env
    
    def update_from_python(self, namespace: Dict[str, Any]):
        """
        Transfer Python namespace variables into R environment
        
        Args:
            namespace: dictionary of Python variables
        """
        if not self.is_initialized:
            return
            
        for key, value in namespace.items():
            try:
                if isinstance(value, pd.DataFrame):
                    # Convert pandas DataFrame to R data frame using pandas2ri
                    self.r_env[key] = self.pandas2ri.py2rpy(value)
                elif isinstance(value, (int, float, str, bool, list)):
                    # Basic types can be converted directly
                    self.r_env[key] = value
                # Extend conversion logic for other types if needed
            except Exception as e:
                print(f"Warning: Failed to convert variable {key} into R environment: {str(e)}")
    
    def update_to_python(self, namespace: Dict[str, Any], variables: Optional[list] = None):
        """
        Retrieve variables from R environment into Python namespace
        
        Args:
            namespace: Python dictionary to receive variables
            variables: list of R variable names to retrieve; if None, retrieve all
        """
        if not self.is_initialized:
            return namespace
            
        if variables is None:
            # Get all variable names from R environment
            try:
                variables = list(self.robjects.r('ls()'))
            except:
                print("Warning: Failed to get variable list from R environment")
                return namespace
        
        for var_name in variables:
            try:
                if var_name in self.r_env:
                    r_value = self.r_env[var_name]
                    
                    # Check if it is a data frame
                    is_dataframe = self.robjects.r(f'is.data.frame({var_name})')[0]
                    
                    if is_dataframe:
                        with self.localconverter(self.robjects.default_converter + self.pandas2ri.converter):
                            namespace[var_name] = self.robjects.conversion.rpy2py(r_value)
                    else:
                        # Try to convert other types
                        namespace[var_name] = r_value
            except Exception as e:
                print(f"Warning: Failed to convert R variable {var_name} into Python: {str(e)}")
        
        return namespace
    
    def execute_r_code(self, code: str, py_namespace: Dict[str, Any] = None) -> Any:
        """
        Execute R code and return results
        
        Args:
            code: R code to execute
            py_namespace: optional Python namespace to share with R
            
        Returns:
            Execution result (if any)
        """
        if not self.is_initialized:
            raise RuntimeError("R environment manager not properly initialized")
            
        # Update R environment with Python namespace if provided
        if py_namespace:
            self.update_from_python(py_namespace)
            
        # Add option settings before execution to prevent interactive prompts
        safe_code = '''
        old_options <- options()
        options(menu.graphics=FALSE)
        options(warn=-1)
        options(interactive=FALSE)
        options(device="png")
        
        IRkernel::installspec()
        ''' + code + '''
        
        options(old_options[!names(old_options) %in% c("warn", "interactive", "device", "menu.graphics")])
        '''

        # Execute R code
        result = self.robjects.r(safe_code)
        # result = self.robjects.r(code)
        
        
        return result

    
    def list_variables(self):
        """
        List all variables in the R environment
        
        Returns:
            List of variable names
        """
        if not self.is_initialized:
            return []
            
        try:
            return list(self.robjects.r('ls()'))
        except:
            return []
    
    def save_environment(self, filepath=".RData"):
        """
        Save current R environment to file
        
        Args:
            filepath: path to save environment
        """
        if not self.is_initialized:
            return
            
        self.robjects.r(f'save.image(file="{filepath}")')
        print(f"R environment saved to: {filepath}")
    
    def load_environment(self, filepath=".RData"):
        """
        Load R environment from file
        
        Args:
            filepath: path to environment file
        """
        if not self.is_initialized:
            return
            
        try:
            self.robjects.r(f'load("{filepath}")')
            print(f"R environment loaded from {filepath}")
        except:
            print(f"Warning: Failed to load R environment from {filepath}")


    def execute_r_code_with_output(self, code, py_namespace=None):
        """Capture output using R's built-in capture.output function"""
        if not self.is_initialized:
            raise RuntimeError("R environment manager not properly initialized")
        
        if py_namespace:
            self.update_from_python(py_namespace)
        
        # Wrap code inside capture.output
        wrapped_code = f'''
        output_capture <- capture.output({{
        {code}
        }}, type = "output")
        
        # Save output for Python access
        .output_captured <- paste(output_capture, collapse="\\n")
        # Return last expression value
        invisible(.output_captured)
        '''
        
        # Execute wrapped code
        result = self.execute_r_code(wrapped_code, py_namespace)
        
        # Retrieve captured output from R environment
        with self.localconverter(self.robjects.default_converter + self.pandas2ri.converter):
            captured_output = self.r_env['.output_captured']
        
        # Clean up temporary variable
        self.execute_r_code('rm(.output_captured)')
        
        return result, captured_output


def enhanced_code_execution(state, memory_manager, r_memory_manager):
    """
    Enhanced execution function supporting both Python and R
    
    Args:
        state: dictionary containing execution state info
        memory_manager: Python memory manager
        r_memory_manager: R environment manager
    
    Returns:
        Execution result dictionary
    """
    messages = state.get("messages", [])
    code_solution = state["code_solution"]
    code = code_solution.code
    lang = state.get("lang", "python")  # use get() to avoid KeyError
    
    stdout_capture = io.StringIO()
    try:
        with warnings.catch_warnings(record=True) as captured_warnings, redirect_stdout(stdout_capture):
            warnings.simplefilter("default")  # Capture all warnings
            
            if lang.lower() == "python":
                # Execute Python code
                local_namespace = memory_manager.get_namespace()
                exec(code, {}, local_namespace)
                memory_manager.update(local_namespace)
            elif lang.lower() == "r":
                # Execute R code
                py_namespace = memory_manager.get_namespace()
                
                # Execute R code and get results
                result = r_memory_manager.execute_r_code(code, py_namespace)
                
                # Update Python namespace with R variables
                updated_namespace = r_memory_manager.update_to_python(py_namespace)
                memory_manager.update(updated_namespace)
                
                # Store R return value into namespace if present
                if result is not None:
                    py_namespace['r_result'] = result
                    memory_manager.update(py_namespace)
            else:
                raise ValueError(f"Unsupported language: {lang}. Supported: 'python' and 'r'")
                
    except Exception as e:
        print(f"---Code check ({lang}): Failed---")
        error_message = traceback.format_exc()
        warning_message = "\n".join([str(warning.message) for warning in captured_warnings])
        error_content = f"Code execution failed:\n{error_message}\nWarnings:\n{warning_message[:500]}\n"
        error_messages = [("user", error_content)]
        messages += error_messages
        
        logger = state.get("logger")
        if logger:
            logger.log_agent_output(error_messages, f"{lang}_code_execution_error", "user")
        
        return {
            "messages": messages,
            "error": 1,
        }

    stdout_output = stdout_capture.getvalue()
    
    logger = state.get("logger")
    if stdout_output:
        if logger:
            logger.log_agent_output(stdout_output[:500], f"{lang}_code_execution_success", "user")
        
        return {
            "messages": messages + [stdout_output],
            "error": 0,            
        }
    else:
        if logger:
            logger.log_agent_output("", f"{lang}_code_execution_success", "user")
        return {
            "error": 0,
        }