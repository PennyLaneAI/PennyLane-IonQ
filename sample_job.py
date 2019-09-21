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
                            },
                            {
                                "gate": "cnot",
                                "target": 1,
                                "control": 0
                            },
                            {
                                "gate": "rx",
                                "target": 0,
                                "rotation": 0.421412
                            }
                        ]
                    }

bell_circuit = \
    {
      "qubits": 2,
      "circuit": [
        {
          "gate": "h",
          "target": 0
        },
        {
          "gate": "cnot",
          "control": 0,
          "target": 1
        }
      ]
    }


sample_job = {"lang": "json",
              "body": sample_circuit}

job = Job()
job.manager.create(**sample_job)
job.manager.get(job.id.value)
print(job.data)
