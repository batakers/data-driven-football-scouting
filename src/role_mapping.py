POSITION_TO_ROLE = {
    "Goalkeeper": "GK",
    "GK": "GK",

    "Centre-Back": "CB",
    "Center-Back": "CB",
    "CB": "CB",
    "Left-Back": "LB",
    "LB": "LB",
    "Right-Back": "RB",
    "RB": "RB",
    "Left Wing-Back": "LWB",
    "LWB": "LWB",
    "Right Wing-Back": "RWB",
    "RWB": "RWB",

    "Defensive Midfield": "CDM",
    "DM": "CDM",
    "CDM": "CDM",
    "Central Midfield": "CM",
    "CM": "CM",
    "Attacking Midfield": "CAM",
    "AM": "CAM",
    "CAM": "CAM",
    "Left Midfield": "LM",
    "LM": "LM",
    "Right Midfield": "RM",
    "RM": "RM",

    "Left Winger": "LW",
    "LW": "LW",
    "Right Winger": "RW",
    "RW": "RW",
    "Second Striker": "CF",
    "SS": "CF",
    "CF": "ST",
    "Centre-Forward": "ST",
    "Center-Forward": "ST",
    "ST": "ST",
}

ROLE_FAMILY = {
    "GK": "goalkeeper",
    "CB": "central_defender",
    "LB": "wide_defender",
    "RB": "wide_defender",
    "LWB": "wide_defender",
    "RWB": "wide_defender",
    "CDM": "defensive_midfielder",
    "CM": "central_midfielder",
    "CAM": "attacking_midfielder",
    "LM": "wide_midfielder",
    "RM": "wide_midfielder",
    "LW": "wide_forward",
    "RW": "wide_forward",
    "CF": "second_striker",
    "ST": "striker",
}

ROLE_SIDE = {
    "LB": "left",
    "LWB": "left",
    "LM": "left",
    "LW": "left",
    "RB": "right",
    "RWB": "right",
    "RM": "right",
    "RW": "right",
    "CB": "central",
    "CDM": "central",
    "CM": "central",
    "CAM": "central",
    "CF": "central",
    "ST": "central",
    "GK": "central",
}

ROLE_COMPATIBILITY = {
    "GK": {"GK": 1.0},
    "ST": {"ST": 1.0, "CF": 0.9, "LW": 0.6, "RW": 0.6},
    "CF": {"CF": 1.0, "ST": 0.9, "CAM": 0.65, "LW": 0.55, "RW": 0.55},
    "LW": {"LW": 1.0, "RW": 0.75, "LM": 0.70, "CAM": 0.60, "ST": 0.60},
    "RW": {"RW": 1.0, "LW": 0.75, "RM": 0.70, "CAM": 0.60, "ST": 0.60},
    "CAM": {"CAM": 1.0, "CM": 0.75, "CF": 0.65, "LW": 0.60, "RW": 0.60},
    "CM": {"CM": 1.0, "CDM": 0.85, "CAM": 0.75, "LM": 0.55, "RM": 0.55},
    "CDM": {"CDM": 1.0, "CM": 0.85, "CB": 0.60},
    "LM": {"LM": 1.0, "LW": 0.75, "LWB": 0.70, "LB": 0.65, "CM": 0.55},
    "RM": {"RM": 1.0, "RW": 0.75, "RWB": 0.70, "RB": 0.65, "CM": 0.55},
    "CB": {"CB": 1.0, "LB": 0.60, "RB": 0.60, "CDM": 0.60},
    "LB": {"LB": 1.0, "LWB": 0.90, "LM": 0.75, "CB": 0.60},
    "RB": {"RB": 1.0, "RWB": 0.90, "RM": 0.75, "CB": 0.60},
    "LWB": {"LWB": 1.0, "LB": 0.90, "LM": 0.85, "LW": 0.60},
    "RWB": {"RWB": 1.0, "RB": 0.90, "RM": 0.85, "RW": 0.60},
}

BROAD_ROLE_GROUP = {
    "GK": "Goalkeeper",
    "CB": "Defender",
    "LB": "Defender",
    "RB": "Defender",
    "LWB": "Defender",
    "RWB": "Defender",
    "CDM": "Midfielder",
    "CM": "Midfielder",
    "CAM": "Midfielder",
    "LM": "Midfielder",
    "RM": "Midfielder",
    "LW": "Forward",
    "RW": "Forward",
    "CF": "Forward",
    "ST": "Forward",
}

COMPATIBLE_ROLE_THRESHOLD = 0.55


def normalize_position(position):
    if position is None:
        return ""
    value = str(position).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return ""
    return value


def map_position_to_role(position):
    value = normalize_position(position)
    if not value:
        return "UNKNOWN"
    return POSITION_TO_ROLE.get(value, POSITION_TO_ROLE.get(value.upper(), "UNKNOWN"))


def unique_roles(*positions):
    roles = []
    for position in positions:
        role = map_position_to_role(position)
        if role != "UNKNOWN" and role not in roles:
            roles.append(role)
    return roles


def compatible_roles(role, threshold=COMPATIBLE_ROLE_THRESHOLD):
    return [
        candidate
        for candidate, score in ROLE_COMPATIBILITY.get(role, {}).items()
        if score >= threshold
    ]


def role_compatibility_score(target_role, candidate_role):
    if target_role == "GK" and candidate_role != "GK":
        return 0.0
    if candidate_role == "GK" and target_role != "GK":
        return 0.0
    return float(ROLE_COMPATIBILITY.get(target_role, {}).get(candidate_role, 0.0))


def foot_role_fit(role, foot):
    if foot is None:
        return 0.90
    foot_value = str(foot).strip().lower()
    if not foot_value or foot_value in {"nan", "none", "null", "unknown"}:
        return 0.90
    if foot_value == "both":
        return 1.00
    if role in {"LB", "LWB", "LM"} and foot_value == "left":
        return 1.00
    if role in {"RB", "RWB", "RM"} and foot_value == "right":
        return 1.00
    if role in {"LB", "LWB", "LM"} and foot_value == "right":
        return 0.80
    if role in {"RB", "RWB", "RM"} and foot_value == "left":
        return 0.80
    if role in {"LW", "RW"}:
        return 0.95
    return 1.00


def side_preference(role):
    return ROLE_SIDE.get(role, "unknown")


def role_family(role):
    return ROLE_FAMILY.get(role, "unknown")
