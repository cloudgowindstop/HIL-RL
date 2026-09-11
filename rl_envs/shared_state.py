"""
本文件仅在rl_env独立使用时使用，不与其他模块共享状态
"""
class SharedState:
    human_intervention_key = False
    success_key = False
    terminate = False
    classifier_apply = True
    classifier = None
    emergency_terminate = False

shared_state = SharedState()