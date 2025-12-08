import json
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from a2a.client.errors import A2AClientJSONRPCError
from a2a.client.middleware import ClientCallContext
from a2a.client.transports import ClientTransport
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    Artifact,
    ContentTypeNotSupportedError,
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    GetTaskPushNotificationConfigParams,
    JSONRPCError,
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)


class AgentInteraction:
    """Configuration container for the agent interaction settings."""

    agent: str | None = None
    dynamic_config: dict[str, Any] | None = None
    deep_research_config: dict[str, Any] | None = None

    def __init__(
        self,
        agent: str | None = None,
        dynamic_config: dict[str, Any] | None = None,
        deep_research_config: dict[str, Any] | None = None,
    ):
        self.agent = agent
        self.dynamic_config = dynamic_config
        self.deep_research_config = deep_research_config

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.agent:
            result['agent'] = self.agent
        if self.dynamic_config:
            result['dynamic_config'] = self.dynamic_config
        if self.deep_research_config:
            result['deep_research_config'] = self.deep_research_config
        return result


# --- Mapping Functions (Dict-based) ---


def _a2a_part_to_interaction_content(part: Part) -> dict[str, Any]:
    """Maps an A2A Part object to an Interactions API Content dictionary."""
    # Access the actual part from the RootModel
    actual_part = part.root

    if isinstance(actual_part, TextPart):
        return {
            'text': {
                'text': actual_part.text,
                'annotations': actual_part.metadata.get('annotations', []) if actual_part.metadata else [],
            }
        }
    elif isinstance(actual_part, FilePart):
        mime_type = None
        blob_content = {}

        if isinstance(actual_part.file, FileWithBytes):
            blob_content['data'] = actual_part.file.bytes
            mime_type = actual_part.file.mime_type
        elif isinstance(actual_part.file, FileWithUri):
            blob_content['uri'] = actual_part.file.uri
            mime_type = actual_part.file.mime_type

        if mime_type:
            blob_content['mime_type'] = mime_type

        if actual_part.metadata:
            resolution = actual_part.metadata.get('resolution')
            if resolution:
                blob_content['resolution'] = resolution

        if mime_type and mime_type.startswith('image'):
            return {'image': blob_content}
        elif mime_type and mime_type.startswith('audio'):
            return {'audio': blob_content}
        elif mime_type and mime_type.startswith('application/pdf'):
            return {'document': blob_content}
        elif mime_type and mime_type.startswith('video'):
            return {'video': blob_content}
        else:
            raise A2AClientJSONRPCError(
                JSONRPCErrorResponse(
                    error=ContentTypeNotSupportedError(
                        message=f'Unsupported file MIME type for Interactions API mapping: {mime_type}'
                    )
                )
            )

    elif isinstance(actual_part, DataPart):
        if 'function_call' in actual_part.data:
            func_call = actual_part.data['function_call']
            return {'function': {'functionCall': {'name': func_call['name'], 'arguments': func_call['args']}}}
        elif 'function_response' in actual_part.data:
            func_response = actual_part.data['function_response']
            # Simplified mapping: stringify the response
            return {
                'functionResponse': {
                    'functionResult': {'stringResult': json.dumps(func_response['response']), 'is_error': False}
                }
            }
        elif 'thought' in actual_part.data:
            thought_data = actual_part.data['thought']
            if isinstance(thought_data, str):
                return {'thought': {'summary': {'items': [{'text': {'text': thought_data}}]}}}
            else:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=ContentTypeNotSupportedError(
                            message='Complex thought DataPart not yet supported for Interactions API mapping.'
                        )
                    )
                )
        else:
            raise A2AClientJSONRPCError(
                JSONRPCErrorResponse(
                    error=ContentTypeNotSupportedError(
                        message=f'Unsupported DataPart type for Interactions API mapping: {actual_part.data.keys()}'
                    )
                )
            )

    raise A2AClientJSONRPCError(
        JSONRPCErrorResponse(
            error=ContentTypeNotSupportedError(
                message=f'Unsupported A2A Part type for Interactions API mapping: {type(actual_part)}'
            )
        )
    )


def _a2a_message_to_interaction_input(message: Message, agent_interaction: AgentInteraction) -> dict[str, Any]:
    """Maps an A2A Message to the input structure for creating an Interaction."""
    interaction_input = {
        'interaction': {
            'contentList': {'contents': [_a2a_part_to_interaction_content(p) for p in message.parts]},
            'previousInteractionId': message.task_id,
        },
        'agentInteraction': agent_interaction.to_dict(),
    }
    return interaction_input


def _interaction_status_to_a2a_task_state(status_str: str) -> TaskState:
    """Maps Interactions API status string to A2A TaskState."""
    mapping = {
        'UNSPECIFIED': TaskState.unknown,
        'IN_PROGRESS': TaskState.working,
        'REQUIRES_ACTION': TaskState.input_required,
        'COMPLETED': TaskState.completed,
        'FAILED': TaskState.failed,
        'CANCELLED': TaskState.canceled,
    }
    return mapping.get(status_str, TaskState.unknown)



def _map_interaction_thought_to_a2a_message(thought_content: dict[str, Any]) -> Message:
    """Maps Interactions API thought content to an A2A Message."""
    thought_parts: list[Part] = []
    summary = thought_content.get('summary', {})
    items = summary.get('items', [])
    for item in items:
        if 'text' in item and 'text' in item['text']:
            thought_parts.append(Part(root=TextPart(text=item['text']['text'])))
        # Add logic for other content types within a thought if needed in the future

    if not thought_parts:
        raise A2AClientJSONRPCError(
            JSONRPCErrorResponse(
                error=ContentTypeNotSupportedError(message=f'Unsupported or empty thought content: {thought_content}')
            )
        )
    return Message(message_id=str(uuid.uuid4()), role=Role.agent, parts=thought_parts)


def _map_interactions_api_content_to_a2a_part(
    content: dict[str, Any],
) -> Part:
    """Maps an Interactions API Content dictionary to an A2A Part object (excluding thoughts)."""
    if 'text' in content:
        text_data = content['text']
        return Part(
            root=TextPart(
                text=text_data.get('text', ''),
                metadata={'annotations': text_data.get('annotations')} if text_data.get('annotations') else {},
            )
        )

    for blob_type in ['image', 'audio', 'document', 'video']:
        if blob_type in content:
            blob_data = content[blob_type]

            mime_type = blob_data.get('mime_type') or f'{blob_type}/unknown'
            if blob_type == 'document' and not blob_data.get('mime_type'):
                mime_type = 'application/octet-stream'

            metadata = {}
            if 'resolution' in blob_data:
                metadata['resolution'] = blob_data['resolution']

            if 'data' in blob_data:
                return Part(
                    root=FilePart(
                        file=FileWithBytes(bytes=blob_data['data'], mime_type=mime_type),
                        metadata=metadata if metadata else None,
                    )
                )
            elif 'uri' in blob_data:
                return Part(
                    root=FilePart(
                        file=FileWithUri(uri=blob_data['uri'], mime_type=mime_type),
                        metadata=metadata if metadata else None,
                    )
                )

    if 'function' in content:
        func_data = content['function']
        # Handles functionCall wrapper if present
        func_call = func_data.get('functionCall', func_data)
        return Part(root=DataPart(data={'function_call': func_call}))

    if 'functionResponse' in content:
        func_res_wrapper = content['functionResponse']
        # Handles functionResult wrapper if present
        func_res_data = func_res_wrapper.get('functionResult', func_res_wrapper)

        if 'contentList' in func_res_data:
            # Stringify content list for now
            contents = func_res_data['contentList'].get('contents', [])
            return Part(root=TextPart(text=f'Function Result: {json.dumps(contents)}'))
        elif 'stringResult' in func_res_data:
            return Part(root=TextPart(text=f'Function Result: {func_res_data["stringResult"]}'))

        return Part(root=DataPart(data={'function_response': func_res_data}))

    raise A2AClientJSONRPCError(
        JSONRPCErrorResponse(
            error=ContentTypeNotSupportedError(
                message=f'Unsupported Interactions API Content type: {list(content.keys())}'
            )
        )
    )

def convert_interaction_to_task(interaction: dict[str, Any]) -> Task:
    """Converts a raw Interactions API dictionary response to an A2A Task."""
    task_id = interaction.get('id', '')

    status_obj = interaction.get('status', {})
    status_str = status_obj.get('status', 'UNSPECIFIED')
    task_status_state = _interaction_status_to_a2a_task_state(status_str)

    task_status_message: Message | None = None

    # Error handling
    error_obj = interaction.get('error') or status_obj.get('error')
    if error_obj:
        task_status_message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[Part(root=TextPart(text=json.dumps(error_obj)))],
        )
    # Check for thoughts in outputs to use as status message
    elif interaction.get('outputs'):
        for output_content in interaction['outputs']:
            if 'thought' in output_content:
                thought = output_content['thought']
                items = thought.get('summary', {}).get('items', [])
                for item in items:
                    if 'text' in item:
                        task_status_message = Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=item['text'].get('text', '')))],
                        )
                        break
                if task_status_message:
                    break

    a2a_artifacts = []
    outputs = interaction.get('outputs', [])
    for output_content in outputs:
        if 'thought' in output_content:
            continue
        try:
            a2a_part = _map_interactions_api_content_to_a2a_part(output_content)
            a2a_artifacts.append(
                Artifact(
                    artifact_id=str(uuid.uuid4()),
                    parts=[a2a_part],
                    name='artifact',
                    metadata={},
                )
            )
        except A2AClientJSONRPCError as e:
            print(f'Warning: Skipping unsupported content type: {e}')
            continue

    a2a_history = []
    turn_list = interaction.get('turnList', {}).get('turns', [])

    for turn in turn_list:
        turn_role = Role.user if turn.get('role') == 'user' else Role.agent
        a2a_message_parts = []

        content_list = turn.get('contentList', {}).get('contents', [])
        if content_list:
            for content in content_list:
                try:
                    a2a_message_parts.append(_map_interactions_api_content_to_a2a_part(content))
                except A2AClientJSONRPCError as e:
                    print(f'Warning: Skipping unsupported history content: {e}')
                    continue
        elif turn.get('contentString'):
            a2a_message_parts.append(Part(root=TextPart(text=turn['contentString'])))

        if a2a_message_parts:
            a2a_history.append(
                Message(
                    message_id=str(uuid.uuid4()),
                    role=turn_role,
                    parts=a2a_message_parts,
                    task_id=task_id,
                )
            )

    return Task(
        id=task_id,
        context_id=task_id,
        status=TaskStatus(state=task_status_state, message=task_status_message),
        artifacts=a2a_artifacts if a2a_artifacts else None,
        history=a2a_history if a2a_history else None,
    )


class InteractionsApiTransport(ClientTransport):
    INTERACTIONS_API_VERSION = 'v1beta'
    EXTENSION_URI = 'https://generativelanguage.googleapis.com/v1beta/a2a'

    _WELL_KNOWN_AGENTS: dict[str, dict[str, Any]] = {
        'deep-research-preview': {
            'deep_research_config': {
                'thinking_summaries': 'THINK_SUMMARIES_AUTO',
            },
        },
    }

    def __init__(self, card: AgentCard, api_key: str | None = None):
        self._card = card
        self._api_key = api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')

        if not self._api_key:
            raise ValueError(
                'API Key not provided and not found in environment variables (GOOGLE_API_KEY or GEMINI_API_KEY).'
            )

        self._base_url = self._card.url
        self._client = httpx.AsyncClient(base_url=self._base_url)
        self._client.headers['x-goog-api-key'] = self._api_key

        # Extract agent_interaction from AgentCard extension
        # We manually look for the extension in capabilities since helpers are not available
        self._agent_interaction = AgentInteraction()
        if self._card.capabilities.extensions:
            for ext in self._card.capabilities.extensions:
                if ext.uri == self.EXTENSION_URI and ext.params:
                    self._agent_interaction = AgentInteraction(
                        agent=ext.params.get('agent'),
                        dynamic_config=ext.params.get('dynamic_config'),
                        deep_research_config=ext.params.get('deep_research_config'),
                    )
                    break

        # Internal state for streaming responses
        self._streaming_tasks: dict[str, Task] = {}

    @classmethod
    def make_card(
        cls,
        url: str,
        agent_name: str,
        agent_config: dict[str, Any] | None = None,
        agent_card_overrides: dict[str, Any] | None = None,
    ) -> AgentCard:
        # Start with well-known agent config if applicable
        final_agent_config = cls._WELL_KNOWN_AGENTS.get(agent_name, {}).copy()
        if agent_config:
            final_agent_config.update(agent_config)

        agent_interaction = AgentInteraction(
            agent=agent_name,
            dynamic_config=final_agent_config.get('dynamic_config'),
            deep_research_config=final_agent_config.get('deep_research_config'),
        )

        ext_params = {}
        if agent_interaction.agent:
            ext_params['agent'] = agent_interaction.agent
        if agent_interaction.dynamic_config:
            ext_params['dynamic_config'] = agent_interaction.dynamic_config
        if agent_interaction.deep_research_config:
            ext_params['deep_research_config'] = agent_interaction.deep_research_config

        # Default AgentCard values
        card_defaults: dict[str, Any] = {
            'url': url,
            'preferred_transport': 'interactions-api',
            'capabilities': AgentCapabilities(
                streaming=True,
                extensions=[
                    AgentExtension(
                        uri=cls.EXTENSION_URI,
                        required=True,
                        params=ext_params,
                    )
                ],
            ),
            'name': f'Interactions API Agent: {agent_name}',
            'description': 'Agent powered by Google Interactions API',
            'default_input_modes': ['text/plain'],
            'default_output_modes': ['text/plain'],
            'skills': [],
            'version': '0.1.0',
        }

        # Apply overrides
        if agent_card_overrides:
            # Handle capabilities and extensions merging if present in overrides
            if 'capabilities' in agent_card_overrides:
                # Create a mutable copy of the default capabilities for merging
                merged_capabilities_dict = (
                    card_defaults['capabilities'].model_dump()  # Use model_dump to get dict from Pydantic model
                )
                override_capabilities = agent_card_overrides['capabilities']

                # Merge extensions: ensure our required extension is maintained
                existing_extensions = merged_capabilities_dict.get('extensions', [])
                override_extensions = override_capabilities.get('extensions', [])

                # Filter out existing extension by our URI before adding it back
                filtered_extensions = (
                    [ext for ext in existing_extensions if ext.get('uri') != cls.EXTENSION_URI]
                    if isinstance(existing_extensions, list)
                    else []
                )

                # Add our generated extension
                our_extension_dict = AgentExtension(
                    uri=cls.EXTENSION_URI,
                    required=True,
                    params=ext_params,
                ).model_dump()
                filtered_extensions.append(our_extension_dict)  # Append our extension as a dict

                # Add other extensions from overrides, avoiding duplicates of our URI
                for ov_ext in override_extensions:
                    if ov_ext.get('uri') != cls.EXTENSION_URI:  # Avoid duplicating our extension
                        filtered_extensions.append(ov_ext)

                override_capabilities['extensions'] = filtered_extensions

                # Update merged_capabilities_dict with override capabilities
                merged_capabilities_dict.update(override_capabilities)
                card_defaults['capabilities'] = AgentCapabilities(**merged_capabilities_dict)

            # Apply other top-level card overrides
            card_defaults.update(agent_card_overrides)

        return AgentCard(**card_defaults)

    @classmethod
    def setup(cls, client_config: Any, client_factory: Any, api_key: str | None = None):
        client_config.add_supported_transport('interactions-api')
        client_factory.register_transport_factory('interactions-api', lambda card: cls(card, api_key))

    async def _make_request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        try:
            headers = {'Content-Type': 'application/json'}
            if stream:
                headers['Accept'] = 'text/event-stream'

            if method == 'POST':
                response = await self._client.post(path, json=json_data, params=params, headers=headers, timeout=None)
            elif method == 'GET':
                response = await self._client.get(path, params=params, headers=headers, timeout=None)
            else:
                raise ValueError(f'Unsupported HTTP method: {method}')

            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=JSONRPCError(code=-32602, message='Invalid parameters', data=e.response.json())
                    )
                ) from e
            elif e.response.status_code == 404:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=JSONRPCError(code=-32001, message='Task not found', data=e.response.json())
                    )
                ) from e
            elif e.response.status_code == 405:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=UnsupportedOperationError(message='Method not supported by Interactions API transport.')
                    )
                ) from e
            else:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=JSONRPCError(code=-32603, message='Internal error', data=e.response.json())
                    )
                ) from e
        except httpx.RequestError as e:
            raise A2AClientJSONRPCError(
                JSONRPCErrorResponse(error=JSONRPCError(code=-32603, message=f'Network or client error: {e}'))
            ) from e

    async def send_message(
        self,
        request: MessageSendParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task | Message:
        interaction_input = _a2a_message_to_interaction_input(request.message, self._agent_interaction)
        payload = {
            'stream': False,
            'store': True,
            'interaction': interaction_input['interaction'],
        }
        # Add root-level params if present in extension config
        if self._agent_interaction.agent:
            payload['agent'] = self._agent_interaction.agent
        if self._agent_interaction.deep_research_config:
            payload['deepResearchConfig'] = self._agent_interaction.deep_research_config
        if self._agent_interaction.dynamic_config:
            payload['dynamicConfig'] = self._agent_interaction.dynamic_config

        response = await self._make_request(
            'POST', f'/{self.INTERACTIONS_API_VERSION}/interactions:create', json_data=payload
        )
        return convert_interaction_to_task(response.json())

    async def _process_stream(
        self,
        response: httpx.Response,
        initial_task: Task | None = None,
    ) -> AsyncGenerator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
        current_task = initial_task
        async for line in response.aiter_lines():
            if line.startswith('data:'):
                try:
                    json_data = json.loads(line[len('data:') :].strip())
                    event_type = json_data.get('eventType')
                    event_data = json_data.get('event', {})

                    if not event_type:
                        continue

                    if event_type == 'InteractionEvent':
                        if 'interaction' in event_data:
                            current_task = convert_interaction_to_task(event_data['interaction'])
                            self._streaming_tasks[current_task.id] = current_task
                            yield current_task
                        elif event_data.get('completed') and current_task:
                            current_task.status.state = TaskState.completed
                            yield TaskStatusUpdateEvent(
                                task_id=current_task.id,
                                context_id=current_task.context_id,
                                status=current_task.status,
                                final=True,
                            )
                            if current_task.id in self._streaming_tasks:
                                del self._streaming_tasks[current_task.id]

                    elif event_type == 'InteractionStatusUpdate' and current_task:
                        status_str = event_data.get('status', 'UNSPECIFIED')
                        current_task.status.state = _interaction_status_to_a2a_task_state(status_str)

                        error_data = event_data.get('error')
                        if error_data:
                            current_task.status.message = Message(
                                message_id=str(uuid.uuid4()),
                                role=Role.agent,
                                parts=[Part(root=TextPart(text=json.dumps(error_data)))],
                            )

                        yield TaskStatusUpdateEvent(
                            task_id=current_task.id,
                            context_id=current_task.context_id,
                            status=current_task.status,
                            final=False,
                        )

                    elif event_type == 'Content' and current_task:
                        if 'thought' in event_data:
                            try:
                                # Convert thought content directly to an A2A Message
                                thought_message = _map_interaction_thought_to_a2a_message(event_data['thought'])
                                current_task.status.message = thought_message
                                yield TaskStatusUpdateEvent(
                                    task_id=current_task.id,
                                    context_id=current_task.context_id,
                                    status=current_task.status,
                                    final=False,
                                )
                            except A2AClientJSONRPCError as e:
                                print(f'Warning: Skipping unsupported thought content in stream: {e}')
                        else:
                            try:
                                a2a_part = _map_interactions_api_content_to_a2a_part(event_data)
                                yield TaskArtifactUpdateEvent(
                                    task_id=current_task.id,
                                    context_id=current_task.context_id,
                                    artifact=Artifact(
                                        artifact_id=str(uuid.uuid4()),
                                        parts=[a2a_part],
                                        name='artifact',
                                        metadata={},
                                    ),
                                    last_chunk=False,
                                )
                            except A2AClientJSONRPCError as e:
                                print(f'Warning: Skipping unsupported content in stream: {e}')

                except json.JSONDecodeError:
                    print(f'Warning: JSON decode error: {line}')

            elif line.strip() == 'event: end' and current_task:
                if current_task.id in self._streaming_tasks:
                    del self._streaming_tasks[current_task.id]
                yield TaskStatusUpdateEvent(
                    task_id=current_task.id,
                    context_id=current_task.context_id,
                    status=current_task.status,
                    final=True,
                )

    async def send_message_streaming(
        self,
        request: MessageSendParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncGenerator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
        interaction_input = _a2a_message_to_interaction_input(request.message, self._agent_interaction)
        payload = {
            'stream': True,
            'store': True,
            'interaction': interaction_input['interaction'],
        }
        # Add root-level params
        if self._agent_interaction.agent:
            payload['agent'] = self._agent_interaction.agent
        if self._agent_interaction.deep_research_config:
            payload['deepResearchConfig'] = self._agent_interaction.deep_research_config
        if self._agent_interaction.dynamic_config:
            payload['dynamicConfig'] = self._agent_interaction.dynamic_config

        async with self._client.stream(
            'POST',
            f'/{self.INTERACTIONS_API_VERSION}/interactions:createStream',
            json=payload,
            headers={'Accept': 'text/event-stream'},
            timeout=None,
        ) as response:
            response.raise_for_status()
            async for event in self._process_stream(response):
                yield event

    async def get_task(
        self,
        request: TaskQueryParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        response = await self._make_request('GET', f'/{self.INTERACTIONS_API_VERSION}/interactions/{request.id}:poll')
        return convert_interaction_to_task(response.json())

    async def cancel_task(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        response = await self._make_request(
            'POST', f'/{self.INTERACTIONS_API_VERSION}/interactions/{request.id}:cancel'
        )
        return convert_interaction_to_task(response.json())

    async def set_task_callback(
        self,
        request: TaskPushNotificationConfig,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> TaskPushNotificationConfig:
        raise A2AClientJSONRPCError(
            JSONRPCErrorResponse(
                error=UnsupportedOperationError(
                    message='Push Notifications are not supported by Interactions API transport.'
                )
            )
        )

    async def get_task_callback(
        self,
        request: GetTaskPushNotificationConfigParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> TaskPushNotificationConfig:
        raise A2AClientJSONRPCError(
            JSONRPCErrorResponse(
                error=UnsupportedOperationError(
                    message='Push Notifications are not supported by Interactions API transport.'
                )
            )
        )

    async def resubscribe(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncGenerator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
        async with self._client.stream(
            'GET',
            f'/{self.INTERACTIONS_API_VERSION}/interactions/{request.id}:stream',
            headers={'Accept': 'text/event-stream'},
            timeout=None,
        ) as response:
            response.raise_for_status()
            current_task: Task | None = self._streaming_tasks.get(request.id)
            if not current_task:
                try:
                    # Sync state first
                    current_task = await self.get_task(TaskQueryParams(id=request.id))
                    self._streaming_tasks[request.id] = current_task
                    yield current_task
                except Exception as e:
                    print(f'Error fetching task for resubscribe: {e}')
                    raise

            async for event in self._process_stream(response, initial_task=current_task):
                yield event

    async def get_card(
        self,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> AgentCard:
        return self._card

    async def close(self) -> None:
        await self._client.aclose()
