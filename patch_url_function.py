# pylint: disable=all
# fmt: off
# flake8: noqa
import aiohttp,os
from datetime import datetime
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)
BACKEND_URL = os.getenv("BACKEND_URL")

async def patch_url_function_progress(url, response_data, uuid=None):
    try:
        logger.info(url)
        # if hasattr(response_data, 'deepsearch_status'):
        #     logger.info(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} URL: {url}\nUUID: {uuid}\nDEEPSEARCH ID/TOKEN/STATUS: {response_data.id}/{response_data.token}/{response_data.deepsearch_status}\nREPORT: {response_data.report}\n")
        # else:
        #     logger.info(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} URL: {url}\nRESPONSE: {response_data.model_dump_json(indent=3)}\n")
        async with aiohttp.ClientSession() as session:
            async with session.patch(url, json=response_data.model_dump(exclude_none=True)) as response:
                if response.status != 200:
                    logger.info(f"Failed with status code: {response.status}")
                    logger.info(await response.text())  # Print error message
                    raise Exception(f"Patch request failed with status code: {response.status}")
        # #     print(response_data)
        logger.info(response_data.model_dump_json(indent=3))
    except Exception as e:
        logger.info(f"An error occurred while patching the URL: URL :{url}\nRESPONSE: {response_data.model_dump_json(indent=3)}\nERROR: {str(e)}")
        raise Exception("patch_url_function_progress --> " + str(e))