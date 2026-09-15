"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 REPRODUCTION CONSTANTS
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Defines the paper targets, Figure 6(c) class information, inferred sample quotas,
    published top-20 features, label aliases, and project-root path used by the GRU CICDDoS2019
    reproduction.

    Key features include:
        - Stores the published Table 4 accuracy and F1 targets.
        - Stores the transcribed Figure 6(c) labels, supports, and confusion matrix.
        - Stores the published top-20 feature list and canonical raw-label aliases.

Usage:
    1. Import constants from other gru_ddos_detection modules.
    2. Do not modify values during runtime.
    3. Use PROJECT_ROOT for behavior that must remain relative to the top-level main.py.

Outputs:
    - None directly produced.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - Python standard library.

Assumptions & Notes:
    - Figure 6(c) class quotas are inferred from the published test support divided by the
      paper's 30% test fraction, exactly as in the supplied monolithic implementation.
================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_TARGET_ACCURACY = 0.9954
PAPER_TARGET_F1 = 0.9800
PAPER_TEST_FRACTION = 0.30

PAPER_FIGURE6_CLASSES: Tuple[str, ...] = (
    "BENIGN",
    "DrDoS_DNS",
    "DrDoS_LDAP",
    "DrDoS_MSSQL",
    "DrDoS_NTP",
    "DrDoS_NetBIOS",
    "DrDoS_SNMP",
    "DrDoS_SSDP",
    "DrDoS_UDP",
    "Syn",
    "TFTP",
    "UDP-lag",
)

PAPER_FIGURE6_TEST_SUPPORT: Dict[str, int] = {
    "BENIGN": 1_476,
    "DrDoS_DNS": 14_584,
    "DrDoS_LDAP": 12_885,
    "DrDoS_MSSQL": 171_644,
    "DrDoS_NTP": 17_988,
    "DrDoS_NetBIOS": 154_321,
    "DrDoS_SNMP": 140_893,
    "DrDoS_SSDP": 15_515,
    "DrDoS_UDP": 18_537,
    "Syn": 12_321,
    "TFTP": 175_669,
    "UDP-lag": 19_922,
}

FIGURE6_INFERRED_CLASS_QUOTAS: Dict[str, int] = {
    class_name: int(round(test_rows / PAPER_TEST_FRACTION))
    for class_name, test_rows in PAPER_FIGURE6_TEST_SUPPORT.items()
}

PAPER_FIGURE6_GRU_CM = np.array(
    [
        [1414, 1, 0, 0, 5, 0, 0, 0, 1, 5, 41, 9],
        [0, 14330, 178, 11, 65, 0, 0, 0, 0, 0, 0, 0],
        [0, 154, 12718, 13, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 9, 892, 170417, 0, 326, 0, 0, 0, 0, 0, 0],
        [0, 51, 0, 0, 17937, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 56, 0, 154122, 142, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 125, 140761, 7, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 246, 15237, 31, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 4, 293, 18149, 0, 0, 90],
        [17, 0, 0, 0, 0, 0, 0, 0, 0, 11921, 99, 284],
        [25, 0, 0, 0, 0, 0, 0, 0, 0, 100, 175544, 0],
        [12, 0, 0, 0, 0, 0, 0, 0, 134, 11, 0, 19765],
    ],
    dtype=np.int64,
)

PAPER_TOP20: Tuple[str, ...] = (
    "Timestamp",
    "Source Port",
    "Min Packet Length",
    "Fwd Packet Length Min",
    "Flow ID",
    "Packet Length Mean",
    "Fwd Packet Length Max",
    "Average Packet Size",
    "ACK Flag Count",
    "Avg Fwd Segment Size",
    "Fwd Packet Length Mean",
    "Flow Bytes/s",
    "Max Packet Length",
    "Protocol",
    "Fwd Packets/s",
    "Flow Packets/s",
    "Total Length of Fwd Packets",
    "Subflow Fwd Bytes",
    "Destination Port",
    "act_data_pkt_fwd",
)

CATEGORICAL_SELECTED = {"Timestamp", "Flow ID"}

BASE_LABEL_ALIASES: Dict[str, str] = {
    "BENIGN": "BENIGN",
    "DRDOSDNS": "DrDoS_DNS",
    "DNS": "DrDoS_DNS",
    "DRDOSLDAP": "DrDoS_LDAP",
    "LDAP": "DrDoS_LDAP",
    "DRDOSMSSQL": "DrDoS_MSSQL",
    "MSSQL": "DrDoS_MSSQL",
    "DRDOSNTP": "DrDoS_NTP",
    "NTP": "DrDoS_NTP",
    "DRDOSNETBIOS": "DrDoS_NetBIOS",
    "NETBIOS": "DrDoS_NetBIOS",
    "DRDOSSNMP": "DrDoS_SNMP",
    "SNMP": "DrDoS_SNMP",
    "DRDOSSSDP": "DrDoS_SSDP",
    "SSDP": "DrDoS_SSDP",
    "DRDOSUDP": "DrDoS_UDP",
    "UDP": "DrDoS_UDP",
    "SYN": "Syn",
    "TFTP": "TFTP",
    "UDPLAG": "UDP-lag",
    "WEBDDOS": "WebDDoS",
    "PORTMAP": "Portmap",
    "PORTSCAN": "Portmap",
}
