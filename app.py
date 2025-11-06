"""
MACM Graph Metrics Microservice

Simple Flask API to calculate graph metrics from MACM files.
"""

from flask import Flask, request, jsonify
import tempfile
import os
from src import calculate_all_metrics_from_cypher

app = Flask(__name__)

# Configuration from environment variables
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/metrics', methods=['POST'])
def calculate_metrics():
    """
    Calculate graph metrics from two MACM files.
    
    Expects:
        - file1: First MACM file (multipart/form-data)
        - file2: Second MACM file (multipart/form-data)
    
    Returns:
        JSON with metrics:
        - edit_distance
        - normalized_edit_distance
        - mcs_size
        - mcs_ratio_graph1
        - mcs_ratio_graph2
    """
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Missing file1 or file2'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Save files temporarily
        with tempfile.NamedTemporaryFile(mode='w', suffix='.macm', delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.macm', delete=False) as tmp2:
            
            tmp1.write(file1.read().decode('utf-8'))
            tmp2.write(file2.read().decode('utf-8'))
            tmp1_path = tmp1.name
            tmp2_path = tmp2.name
        
        # Calculate metrics
        metrics = calculate_all_metrics_from_cypher(
            tmp1_path,
            tmp2_path,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD
        )
        
        # Cleanup
        os.unlink(tmp1_path)
        os.unlink(tmp2_path)
        
        return jsonify({
            'edit_distance': metrics.edit_distance,
            'normalized_edit_distance': metrics.normalized_edit_distance,
            'mcs_size': metrics.mcs_size,
            'mcs_ratio_graph1': metrics.mcs_ratio_graph1,
            'mcs_ratio_graph2': metrics.mcs_ratio_graph2
        }), 200
        
    except Exception as e:
        # Cleanup on error
        try:
            os.unlink(tmp1_path)
            os.unlink(tmp2_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
