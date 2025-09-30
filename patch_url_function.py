from datetime import datetime
import aiohttp


async def patch_url_function_progress(url, response_data, uuid=None):
    """
    Async function to send PATCH requests to a URL with response data
    
    Args:
        url (str): The URL to send the PATCH request to
        response_data: Pydantic model or dict with data to send
        uuid (str, optional): UUID for tracking/logging purposes
    
    Returns:
        dict: Response data from the PATCH request
    
    Raises:
        Exception: If the PATCH request fails
    """
    try:
        print(
            f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} URL: {url}\nUUID: {uuid}\nRESPONSE: {response_data.model_dump_json(indent=3) if hasattr(response_data, 'model_dump_json') else str(response_data)}\n"
        )

        async with aiohttp.ClientSession() as session:
            async with session.patch(url, json=response_data.model_dump() if hasattr(response_data, 'model_dump') else response_data) as response:
                if response.status != 200:
                    print(f"Failed with status code: {response.status}")
                    response_text = await response.text()
                    print(response_text)
                    raise Exception(f"Patch request failed with status code: {response.status}")
                
                # Return the response data
                try:
                    return await response.json()
                except:
                    return {"status": "success", "message": "PATCH request completed successfully"}
                    
    except Exception as e:
        print(
            f"An error occurred while patching the URL: URL: {url}\nRESPONSE: {response_data.model_dump_json(indent=3) if hasattr(response_data, 'model_dump_json') else str(response_data)}\nERROR: {str(e)}"
        )
        raise Exception("patch_url_function_progress --> " + str(e))
