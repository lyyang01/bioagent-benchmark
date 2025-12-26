from typing import Dict, Optional, Union, List

from autogen import Agent, AssistantAgent, UserProxyAgent
import chainlit as cl
import os
import autogen
from autogen.coding.jupyter import LocalJupyterServer
from autogen.coding.jupyter import JupyterCodeExecutor, JupyterConnectionInfo

async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res

class ChainlitAssistantAgent(AssistantAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}":*\n\n{message}',
                author=self.name,
            ).send()
        )
        super(ChainlitAssistantAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

class ChainlitAssistantAgent_ex(AssistantAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        #cl.run_sync(
        #    cl.Message(
        #        content=f'*Sending message to "{recipient.name}":*\n\n{message}',
        #        author=self.name,
        #    ).send()
        #)
        super(ChainlitAssistantAgent_ex, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


class ChainlitUserProxyAgent(UserProxyAgent):
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            #"Provide feedback to " + self.name + ". Press enter to skip and use auto-reply"
            "Provide feedback to"
        ) or prompt.startswith(
            #"Provide feedback to " + self.name + ". Press enter to skip and use auto-reply"
            "Please give feedback to"
        ):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or provide feedback?",
                    actions=[
                        cl.Action(
                            name="continue", value="exit", label="✅ Continue"
                        ),
                        cl.Action(
                            name="feedback",
                            value="feedback",
                            label="💬 Provide feedback",
                        ),
                        #cl.Action( 
                        #    name="exit",
                        #    value="exit", 
                        #    label="🔚 Exit Conversation" 
                        #),
                    ],
                    timeout=1000
                )
            )

            #if (res is None):
            #    cl.run_sync(ask_helper(cl.Message, content="🔔🔔🔔✅ AI take over the control to continue..."))
            #    return ""
            if (res is None):
                cl.run_sync(ask_helper(cl.Message, content="🔔🔔🔔✅ AI take over the control to continue..."))
                return ""
            #if res.get("value") == "continue":
            #    self._human_input.append("")
            #    return ""
            #if res.get("value") == "continue":
            #    self._human_input.append("exit")
            #    return "exit"
            if res.get("value") == "exit":
                self._human_input.append("exit")
                return "exit"

        reply = cl.run_sync(ask_helper(cl.AskUserMessage, content=prompt, timeout=1000))
        if reply is None:
            return ""

        #return reply["content"].strip()
        return reply["output"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        try:
            if message.startswith('the user task need') or (message.startswith('{') and message.endswith('}')):
                pass
            else:
                cl.run_sync(
                    cl.Message(
                        content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                        author=self.name,
                    ).send()
                )
        except:
            cl.run_sync(
                    cl.Message(
                        content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                        author=self.name,
                    ).send()
                )

        super(ChainlitUserProxyAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


class ChainlitUserProxyAgent_new(UserProxyAgent):
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            #"Provide feedback to " + self.name + ". Press enter to skip and use auto-reply"
            "Provide feedback to"
        ) or prompt.startswith(
            #"Provide feedback to " + self.name + ". Press enter to skip and use auto-reply"
            "Please give feedback to"
        ):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or provide feedback?",
                    actions=[
                        cl.Action(
                            name="continue", value="exit", label="✅ Continue"
                        ),
                        cl.Action(
                            name="feedback",
                            value="feedback",
                            label="💬 Provide feedback",
                        ),
                        #cl.Action( 
                        #    name="exit",
                        #    value="exit", 
                        #    label="🔚 Exit Conversation" 
                        #),
                    ],
                    timeout=1000
                )
            )

            #if (res is None):
            #    cl.run_sync(ask_helper(cl.Message, content="🔔🔔🔔✅ AI take over the control to continue..."))
            #    return ""
            if (res is None):
                cl.run_sync(ask_helper(cl.Message, content="🔔🔔🔔✅ AI take over the control to continue..."))
                return ""
            if res.get("value") == "feedback":
                self._human_input.append("")
            #if res.get("value") == "continue":
            #    self._human_input.append("")
            #    return ""
            
            #if res.get("value") == "continue":
            #    self._human_input.append("psudo-exit")
            #    return ""
            if res.get("value") == "exit":
                self._human_input.append("psudo-exit")
                return ""

        reply = cl.run_sync(ask_helper(cl.AskUserMessage, content=prompt, timeout=1000))
        if reply is None:
            return ""

        #return reply["content"].strip()
        return reply["output"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        try:
            if message.startswith('the user task need') or (message.startswith('{') and message.endswith('}')):
                pass
            else:
                cl.run_sync(
                    cl.Message(
                        content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                        author=self.name,
                    ).send()
                )
        except:
            cl.run_sync(
                    cl.Message(
                        content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                        author=self.name,
                    ).send()
                )

        super(ChainlitUserProxyAgent_new, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


class ChainlitUserProxyAgent_skip(UserProxyAgent):
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            #"Provide feedback to " + self.name + ". Press enter to skip and use auto-reply"
            "Provide feedback to"
        ) or prompt.startswith(
            #"Provide feedback to " + self.name + ". Press enter to skip and use auto-reply"
            "Please give feedback to"
        ):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or provide feedback?",
                    actions=[
                        cl.Action(
                            name="continue", value="exit", label="✅ No Skip"
                        ),
                        #cl.Action(
                        #    name="feedback",
                        #    value="feedback",
                        #    label="💬 Provide feedback",
                        #),
                        cl.Action(
                            name="skip-step",
                            value="exit", 
                            label="🔚 Skip Step" 
                        ),
                        cl.Action( 
                            name="skip-replan",
                            value="exit", 
                            label="🔚 Skip Re-Plan" 
                        ),
                    ],
                    timeout=1000
                )
            )

            #if (res is None):
            #    cl.run_sync(ask_helper(cl.Message, content="🔔🔔🔔✅ AI take over the control to continue..."))
            #    return ""
            if (res is None):
                cl.run_sync(ask_helper(cl.Message, content="🔔🔔🔔✅ AI take over the control to continue..."))
                return ""
            #if res.get("value") == "feedback":
            #    self._human_input.append("")
            #if res.get("value") == "continue":
            #    self._human_input.append("")
            #    return ""
            
            #if res.get("value") == "continue":
            #    self._human_input.append("psudo-exit")
            #    return ""
            if res.get("value") == "exit":
                self._human_input.append("no-skip")
                return "exit"
            if res.get("value") == "skip-step":
                self._human_input.append("skip-step")
                return "exit"
            if res.get("value") == "skip-replan":
                self._human_input.append("skip-replan")
                return "exit"

            #if res.get("value") == "exit":
            #    self._human_input.append("exit")
            #    return "exit"

        reply = cl.run_sync(ask_helper(cl.AskUserMessage, content=prompt, timeout=1000))
        if reply is None:
            return ""

        #return reply["content"].strip()
        return reply["output"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        try:
            if message.startswith('the user task need') or (message.startswith('{') and message.endswith('}')):
                pass
            else:
                cl.run_sync(
                    cl.Message(
                        content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                        author=self.name,
                    ).send()
                )
        except:
            cl.run_sync(
                    cl.Message(
                        content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                        author=self.name,
                    ).send()
                )

        super(ChainlitUserProxyAgent_skip, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )
