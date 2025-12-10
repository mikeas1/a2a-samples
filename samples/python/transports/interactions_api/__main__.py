import logging
import os
import sys

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from dotenv import load_dotenv

from interactions_api_transport import InteractionsApiTransport
from request_handler import InteractionsAPIProxyRequestHandler

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host, port):
    """Starts the Interactions API proxy server."""
    try:
        if not os.getenv('GOOGLE_API_KEY') and not os.getenv('GEMINI_API_KEY'):
            raise MissingAPIKeyError('both GOOGLE_API_KEY and GEMINI_API_KEY environment variables not set.')

        interactions_agent_card = InteractionsApiTransport.make_card(
            url='https://preprod-generativelanguage.googleapis.com',
            model='gemini-3-pro-preview',
            request_opts={
                # 'generation_config': {'thinking_summaries': 'none'},
            },
            # agent_card_overrides={
            #     'skills': [
            #         AgentSkill(
            #             id='completion',
            #             name='Completion',
            #             description='Get completions for a given request',
            #             examples=['Hello'],
            #             input_modes=['text/plain'],
            #             output_modes=['text/plain'],
            #             tags=['completion'],
            #         )
            #     ]
            # },
        )

        interaction_api_transport_object = InteractionsApiTransport(card=interactions_agent_card)

        request_handler = InteractionsAPIProxyRequestHandler(transport=interaction_api_transport_object)
        exported_card = interactions_agent_card.model_copy(
            update={
                'url': f'http://{host}:{port}',
                'preferred_transport': 'jsonrpc',
            }
        )

        server = A2AStarletteApplication(agent_card=exported_card, http_handler=request_handler)

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
