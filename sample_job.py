from pennylane_ionq.api_client import *

sample_circuit =   {
                        "qubits": 4,
                        "circuit": [
                            {
                                "gate": "h",
                                "target": 3
                            },
                            {
                                "gate": "rx",
                                "target": 0,
                                "rotation": 1
                            },
                            {
                                "gate": "cnot",
                                "target": 2,
                                "control": 0
                            }
                        ]
                    }

sample_job = {"lang": "json",
              "body": sample_circuit}

job = Job()
job.manager.create(body=sample_circuit, lang="json")