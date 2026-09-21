"""
fuzzy_logic.py
Deterministic Mamdani Fuzzy Inference System built with scikit-fuzzy.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def build_fuzzy_system():
    input_universe = np.arange(0, 10.1, 0.1)
    output_universe = np.arange(0, 100.1, 0.5)

    urgency = ctrl.Antecedent(input_universe, 'urgency')
    link_suspicion = ctrl.Antecedent(input_universe, 'link_suspicion')
    sensitive_request = ctrl.Antecedent(input_universe, 'sensitive_request')
    threat_reward = ctrl.Antecedent(input_universe, 'threat_reward')

    scam_risk = ctrl.Consequent(output_universe, 'scam_risk')

    for var in [urgency, link_suspicion, sensitive_request, threat_reward]:
        var['low'] = fuzz.trapmf(var.universe, [0.0, 0.0, 2.0, 4.5])
        var['medium'] = fuzz.trimf(var.universe, [3.0, 5.0, 7.5])
        var['high'] = fuzz.trapmf(var.universe, [6.0, 8.0, 10.0, 10.0])

    scam_risk['safe'] = fuzz.trapmf(scam_risk.universe, [0.0, 0.0, 20.0, 40.0])
    scam_risk['suspicious'] = fuzz.trimf(scam_risk.universe, [30.0, 55.0, 75.0])
    scam_risk['high_risk'] = fuzz.trapmf(scam_risk.universe, [65.0, 85.0, 100.0, 100.0])

    rule1 = ctrl.Rule(urgency['high'] & sensitive_request['high'], scam_risk['high_risk'])
    rule2 = ctrl.Rule(link_suspicion['high'] & sensitive_request['high'], scam_risk['high_risk'])
    rule3 = ctrl.Rule(threat_reward['high'] & (urgency['high'] | sensitive_request['medium']), scam_risk['high_risk'])
    rule4 = ctrl.Rule(link_suspicion['high'] & urgency['medium'], scam_risk['high_risk'])
    rule5 = ctrl.Rule(link_suspicion['medium'] & sensitive_request['medium'], scam_risk['suspicious'])
    rule6 = ctrl.Rule(urgency['medium'] & threat_reward['medium'], scam_risk['suspicious'])
    rule7 = ctrl.Rule(link_suspicion['low'] & sensitive_request['low'] & urgency['low'] & threat_reward['low'], scam_risk['safe'])
    rule8 = ctrl.Rule(threat_reward['high'] & sensitive_request['low'] & link_suspicion['low'], scam_risk['suspicious'])
    rule9 = ctrl.Rule(urgency['high'] & link_suspicion['low'] & sensitive_request['low'], scam_risk['suspicious'])
    rule10 = ctrl.Rule(sensitive_request['high'] | threat_reward['high'], scam_risk['high_risk'])

    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10]
    scam_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(scam_ctrl)

fuzzy_simulator = build_fuzzy_system()

def calculate_scam_risk(urgency_val: float, link_val: float, sensitive_val: float, threat_reward_val: float) -> float:
    u = float(np.clip(urgency_val, 0.0, 10.0))
    l = float(np.clip(link_val, 0.0, 10.0))
    s = float(np.clip(sensitive_val, 0.0, 10.0))
    t = float(np.clip(threat_reward_val, 0.0, 10.0))

    fuzzy_simulator.input['urgency'] = u
    fuzzy_simulator.input['link_suspicion'] = l
    fuzzy_simulator.input['sensitive_request'] = s
    fuzzy_simulator.input['threat_reward'] = t

    try:
        fuzzy_simulator.compute()
        risk_score = fuzzy_simulator.output['scam_risk']
    except Exception:
        risk_score = (u * 0.25 + l * 0.30 + s * 0.30 + t * 0.15) * 10.0

    return round(float(risk_score), 2)
