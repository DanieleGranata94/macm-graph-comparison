"""
Test script for MACM microservice
"""

import requests

# Test health endpoint
print("Testing /health endpoint...")
response = requests.get('http://localhost:5000/health')
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test metrics endpoint
print("Testing /metrics endpoint...")
files = {
    'file1': open('data/macm_files/Ewelink_correct.macm', 'rb'),
    'file2': open('data/macm_files/Ewelink_incorrect.macm', 'rb')
}

response = requests.post('http://localhost:5000/metrics', files=files)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
