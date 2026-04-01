"""
Rate limiting configuration using SlowAPI
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from config import settings

# Initialize limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit_general]
)

# Rate limit decorators
def limit_predict():
    """Rate limit for prediction endpoints"""
    return limiter.limit(settings.rate_limit_predict)

def limit_batch():
    """Rate limit for batch endpoints"""
    return limiter.limit(settings.rate_limit_batch)

def limit_general():
    """Rate limit for general endpoints"""
    return limiter.limit(settings.rate_limit_general)
