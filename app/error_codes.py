"""
Centralized error code definitions
"""

class ErrorCodes:
    """Error code constants and messages"""
    
    # Client errors (4xx)
    INVALID_INPUT = "ERR_1001"
    MISSING_FIELD = "ERR_1002"
    VALIDATION_FAILED = "ERR_1003"
    RATE_LIMIT_EXCEEDED = "ERR_1004"
    INVALID_CUSTOMER_ID = "ERR_1005"
    
    # Server errors (5xx)
    MODEL_NOT_LOADED = "ERR_5001"
    PREDICTION_FAILED = "ERR_5002"
    INTERNAL_ERROR = "ERR_5003"
    DATABASE_ERROR = "ERR_5004"
    MODEL_TRAINING_FAILED = "ERR_5005"
    
    @staticmethod
    def get_message(code: str) -> str:
        """Get human-readable message for error code"""
        messages = {
            # Client errors
            "ERR_1001": "Invalid input data format",
            "ERR_1002": "Required field is missing",
            "ERR_1003": "Data validation failed",
            "ERR_1004": "Rate limit exceeded - too many requests",
            "ERR_1005": "Invalid customer ID",
            
            # Server errors
            "ERR_5001": "Model not loaded or unavailable",
            "ERR_5002": "Prediction generation failed",
            "ERR_5003": "Internal server error",
            "ERR_5004": "Database operation failed",
            "ERR_5005": "Model training failed",
        }
        return messages.get(code, "Unknown error")
    
    @staticmethod
    def get_http_status(code: str) -> int:
        """Get HTTP status code for error code"""
        if code.startswith("ERR_1"):
            return 400  # Bad Request
        elif code.startswith("ERR_5"):
            return 500  # Internal Server Error
        return 500
