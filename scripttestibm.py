from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

service = QiskitRuntimeService()


for backend in service.backends():
    if not backend.simulator:
        print(f"{backend.name} ({backend.num_qubits} qubits)")


#backend = service.least_busy(operational=True, simulator=False)

#qc = QuantumCircuit(2)
#qc.measure_all()

#sampler = Sampler(backend)
#job = sampler.run([qc])
#print(f"job id: {job.job_id()}")
#result = job.result()
#print(result)
