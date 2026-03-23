"""Seed script: ingest source paper (confidence 0.52) + initial DTC papers.

Run: python scripts/seed_papers.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime


async def seed():
    from src.db import init_tables, AsyncSessionLocal
    from src.models import Paper, Hypothesis, Protocol
    from sqlalchemy import select, func

    await init_tables()

    async with AsyncSessionLocal() as session:
        count = (await session.execute(select(func.count(Paper.id)))).scalar()
        if count and count > 0:
            print(f"Already seeded ({count} papers). Skipping.")
            return

        papers = [
            Paper(
                titulo="Discrete time crystal acts as a usable sensor for weak magnetic oscillations",
                autores="DTC Research Group",
                fecha=datetime(2024, 1, 15),
                tipo="EXPERIMENTAL",
                abstract="We demonstrate the first practical use of discrete time crystals as sensors for weak magnetic oscillations. Our DTC-based sensor achieves sensitivity to biomagnetic and geophysical field oscillations at the ~200 pT/sqrt(Hz) level at 10 Hz using NV centers in diamond.",
                parametros_json={
                    "material": "NV centers in diamond", "tipo_material": "NV_DIAMOND",
                    "temperatura_K": 300.0, "sensibilidad_pT": 200.0, "frecuencia_Hz": 10.0,
                    "SNR": 3.5, "T2_us": 1.0, "n_qubits": 15, "geometria_red": "diamond lattice",
                    "driving_tipo": "MICROWAVE", "driving_frecuencia": 2.87e9, "driving_potencia": 0.05,
                    "condiciones_ambientales": "room temperature, ambient shielding",
                },
                reproducibilidad=0.7, novedad=0.9, relevancia_biosensado=0.85, confianza_fuente=0.52,
            ),
            Paper(
                titulo="Floquet time crystal phase in trapped-ion quantum simulator for magnetometry",
                autores="Ion Trap Quantum Lab",
                fecha=datetime(2024, 3, 20), tipo="EXPERIMENTAL",
                abstract="Demonstration of Floquet DTC phase in chain of Yb+ trapped ions with application to magnetic field sensing at low frequencies.",
                parametros_json={
                    "material": "Yb+ trapped ions", "tipo_material": "TRAPPED_ION",
                    "temperatura_K": 0.001, "sensibilidad_pT": 50.0, "frecuencia_Hz": 5.0,
                    "T2_us": 1000.0, "n_qubits": 20, "driving_tipo": "LASER",
                },
                reproducibilidad=0.6, novedad=0.8, relevancia_biosensado=0.7, confianza_fuente=0.55,
            ),
            Paper(
                titulo="Theoretical limits of DTC-enhanced magnetic sensing in NV diamond arrays",
                autores="Quantum Sensing Theory Group",
                fecha=datetime(2024, 5, 10), tipo="TEORICO",
                abstract="We derive theoretical sensitivity bounds for DTC-based magnetometers using dense NV center ensembles, showing potential to reach sub-pT sensitivity.",
                parametros_json={
                    "material": "Dense NV diamond ensemble", "tipo_material": "NV_DIAMOND",
                    "temperatura_K": 300.0, "sensibilidad_pT": 0.5, "n_qubits": 1000, "driving_tipo": "MICROWAVE",
                },
                reproducibilidad=0.5, novedad=0.7, relevancia_biosensado=0.9, confianza_fuente=0.45,
            ),
            Paper(
                titulo="Superconducting DTC for ultra-sensitive biomagnetic detection at millikelvin temperatures",
                autores="Cryogenic Quantum Sensors Lab",
                fecha=datetime(2024, 2, 28), tipo="EXPERIMENTAL",
                abstract="We implement a DTC in a superconducting transmon array and demonstrate magnetic sensitivity approaching SQUID levels.",
                parametros_json={
                    "material": "Transmon array", "tipo_material": "SUPERCONDUCTOR",
                    "temperatura_K": 0.015, "sensibilidad_pT": 0.01, "SNR": 10.0,
                    "T2_us": 50.0, "n_qubits": 8, "driving_tipo": "MICROWAVE",
                },
                reproducibilidad=0.8, novedad=0.6, relevancia_biosensado=0.5, confianza_fuente=0.6,
            ),
            Paper(
                titulo="Room-temperature DTC observation in NV diamond via optical driving",
                autores="Optical Quantum Lab",
                fecha=datetime(2024, 7, 5), tipo="EXPERIMENTAL",
                abstract="First observation of stable DTC order parameter at room temperature using optically driven NV centers.",
                parametros_json={
                    "material": "NV diamond bulk crystal", "tipo_material": "NV_DIAMOND",
                    "temperatura_K": 295.0, "sensibilidad_pT": 500.0, "frecuencia_Hz": 20.0,
                    "T2_us": 0.5, "n_qubits": 50, "driving_tipo": "LASER",
                },
                reproducibilidad=0.65, novedad=0.85, relevancia_biosensado=0.8, confianza_fuente=0.5,
            ),
            Paper(
                titulo="Noise spectroscopy with discrete time crystals: biomagnetic applications",
                autores="Quantum Bio-sensing Institute",
                fecha=datetime(2024, 8, 12), tipo="EXPERIMENTAL",
                abstract="We use DTC subharmonic response for noise spectroscopy and demonstrate detection of 50 pT oscillating fields mimicking cortical neural activity.",
                parametros_json={
                    "material": "NV diamond array", "tipo_material": "NV_DIAMOND",
                    "temperatura_K": 300.0, "sensibilidad_pT": 50.0, "frecuencia_Hz": 10.0,
                    "SNR": 3.2, "T2_us": 2.0, "n_qubits": 30, "driving_tipo": "MICROWAVE",
                },
                reproducibilidad=0.75, novedad=0.75, relevancia_biosensado=0.95, confianza_fuente=0.58,
            ),
            Paper(
                titulo="Bennett-Brassard quantum key distribution applied to sensor data integrity",
                autores="Quantum Cryptography Lab",
                fecha=datetime(2024, 4, 1), tipo="TEORICO",
                abstract="Framework for applying BB84-style quantum verification to validate integrity of quantum sensor data against classical interference.",
                parametros_json={"material": "N/A - protocol", "tipo_material": "OTHER"},
                reproducibilidad=0.8, novedad=0.7, relevancia_biosensado=0.6, confianza_fuente=0.74,
            ),
            Paper(
                titulo="Microwave quantum network resilience: thermal noise characterization",
                autores="Quantum Network Group",
                fecha=datetime(2024, 6, 18), tipo="EXPERIMENTAL",
                abstract="Characterization of thermal resilience in microwave quantum networks relevant to DTC sensor readout.",
                parametros_json={
                    "material": "Microwave quantum network", "tipo_material": "SUPERCONDUCTOR",
                    "temperatura_K": 4.0, "T2_us": 20.0, "driving_tipo": "MICROWAVE",
                },
                reproducibilidad=0.7, novedad=0.5, relevancia_biosensado=0.4, confianza_fuente=0.52,
            ),
        ]

        for p in papers:
            session.add(p)
            print(f"  + {p.titulo[:70]}... (conf: {p.confianza_fuente})")

        # Pilot hypothesis
        session.add(Hypothesis(
            enunciado="Si se implementa una cadena de 10-20 iones Yb+ en trampa de Paul con driving DTC laser 369.5 nm, entonces se detectaran campos de 50 pT a 10 Hz con SNR > 3",
            accion="Implementar cadena 10-20 Yb+ en Paul trap con driving DTC a 369.5 nm",
            resultado_esperado="SNR > 3 a 50 pT, 10 Hz",
            mecanismo="DTC subharmonic amplification of magnetic signal via Floquet phase",
            tipo="COMBINATORIA", confianza=0.45, impacto=0.9, costo_testeo=0.7,
            rank_score=0.45 * 0.9 / 0.7,
        ))
        print("  + Pilot hypothesis seeded")

        # Pilot protocol
        session.add(Protocol(
            titulo="Pilot Protocol: 50 pT Detection at 10 Hz with Trapped Ion DTC",
            objetivo="Detect 50 pT magnetic fields at 10 Hz using DTC of trapped Yb+ ions",
            material="10-20 Yb+ ions in linear Paul trap",
            material_type="TRAPPED_ION",
            sensor_config="Chain of 10-20 Yb+171 ions, axial freq 1 MHz, radial 5 MHz",
            driving_config="Pulsed 369.5 nm laser (S1/2-P1/2), 10us pulses, 100 kHz rep rate",
            detection_method="Time-resolved fluorescence at 369.5 nm -> FFT -> subharmonic peak",
            pasos=["Prepare Paul trap: load 15 Yb+171 ions", "Calibrate Helmholtz coils: 10 Hz, 50 pT",
                   "Initialize DTC driving", "Verify subharmonic in FFT", "Apply test field",
                   "Acquire 100 driving periods", "Calculate SNR", "Repeat at 200 pT",
                   "Integrity verification"],
            metricas_exito={"exito": "SNR>3 at 50pT", "parcial": "SNR>3 at 200pT"},
            seguridad=["Laser safety: Class 3B UV", "Vacuum UHV protocols", "RF high voltage"],
            sensibilidad_predicha_pt=50.0, temperatura_k=0.001, costo_estimado_usd=35000.0,
            confianza=0.45,
        ))
        print("  + Pilot protocol seeded")

        await session.commit()
        print(f"\nSeed complete: {len(papers)} papers, 1 hypothesis, 1 protocol")


if __name__ == "__main__":
    asyncio.run(seed())
