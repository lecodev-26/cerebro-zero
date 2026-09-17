"""
Tests del dashboard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
from dashboard.app import app, state, DashboardState


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_state():
    """Limpia el historial antes de cada test"""
    state.messages.clear()
    yield
    state.messages.clear()


# ============================================
# TESTS DEL ESTADO
# ============================================

def test_state_creation():
    s = DashboardState()
    assert s.agent is not None
    assert s.uptime() >= 0


def test_state_get_stats():
    stats = state.get_stats()
    assert 'uptime_seconds' in stats
    assert 'memory' in stats
    assert 'tools' in stats
    assert 'experiences' in stats


def test_state_process():
    result = state.process("hola")
    assert 'response' in result
    assert 'stats' in result
    assert len(state.messages) == 2  # user + assistant


def test_state_memory_grows():
    initial = state.get_stats()['history_length']
    state.process("5 + 3")
    assert state.get_stats()['history_length'] > initial


# ============================================
# TESTS DE LA API
# ============================================

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Cerebro Zero' in response.data


def test_api_chat_empty(client):
    response = client.post('/api/chat',
                          data=json.dumps({'message': ''}),
                          content_type='application/json')
    assert response.status_code == 400


def test_api_chat_math(client):
    response = client.post('/api/chat',
                          data=json.dumps({'message': '¿Cuánto es 5 + 3?'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert '8' in data['response']


def test_api_chat_hello(client):
    response = client.post('/api/chat',
                          data=json.dumps({'message': 'hola'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert 'Hola' in data['response'] or 'hola' in data['response'].lower()


def test_api_stats(client):
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = response.get_json()
    assert 'uptime_seconds' in data
    assert 'memory' in data


def test_api_history(client):
    client.post('/api/chat',
               data=json.dumps({'message': 'test'}),
               content_type='application/json')
    
    response = client.get('/api/history')
    assert response.status_code == 200
    data = response.get_json()
    assert data['count'] >= 2


def test_api_reset(client):
    # Enviar un mensaje
    client.post('/api/chat',
               data=json.dumps({'message': 'test'}),
               content_type='application/json')
    
    # Limpiar
    response = client.post('/api/reset')
    assert response.status_code == 200
    
    # Verificar
    history = client.get('/api/history').get_json()
    assert history['count'] == 0


def test_api_monitor(client):
    response = client.get('/api/monitor')
    assert response.status_code == 200
    data = response.get_json()
    assert 'ram_mb' in data


def test_api_404(client):
    response = client.get('/nonexistent')
    assert response.status_code == 404


# ============================================
# TESTS DE INTEGRACIÓN
# ============================================

def test_full_conversation(client):
    """Conversación completa"""
    messages = ['hola', '¿Cuánto es 5 + 3?', '¿Qué hora es?', 'adiós']
    
    for msg in messages:
        response = client.post('/api/chat',
                              data=json.dumps({'message': msg}),
                              content_type='application/json')
        assert response.status_code == 200
        data = response.get_json()
        assert 'response' in data
        assert len(data['response']) > 0
    
    # 4 mensajes × 2 (user + assistant) = 8
    history = client.get('/api/history').get_json()
    assert history['count'] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
