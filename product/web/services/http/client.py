import requests
from typing import Dict, Any

from services.config.config import WebBackendConfig
from services.logger.logger import logger


class SyncWebBackendClient:
    def __init__(self, config: WebBackendConfig):
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({
            'accept': 'application/json',
            'Content-Type': 'application/json'
        })
        self._timeout = config.timeout
    
    def send_recommendation_request(self, user_id: str) -> Dict[str, Any]:
        url = f"{self._config.base_url}/recsys/send/{user_id}"
        
        try:
            response = self._session.get(url, timeout=self._timeout)
            return self._handle_response(response)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for user {user_id}: {e}")
            return {"status": "error", "message": f"Request failed: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error for user {user_id}: {e}")
            return {"status": "error", "message": f"Unexpected error: {e}"}
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Обработка ответа от сервера"""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            logger.warning(f"Endpoint not found: {response.text}")
            return {"status": "error", "code": 404, "message": "Endpoint not found"}
        elif response.status_code == 500:
            logger.error(f"Server error: {response.text}")
            return {"status": "error", "code": 500, "message": "Internal server error"}
        else:
            logger.warning(f"Unexpected status {response.status_code}: {response.text}")
            return {
                "status": "error", 
                "code": response.status_code, 
                "message": f"Unexpected status: {response.status_code}"
            }
    
    def close(self):
        """Закрытие сессии"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()