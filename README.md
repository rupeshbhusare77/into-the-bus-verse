


# 🚌 Into the Bus-Verse: Radhanagari ST Depot Digital Twin

**Uncovering Spatiotemporal Transit Failures Through Behavioral Modeling and Queueing Theory**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![SimPy](https://img.shields.io/badge/SimPy-Discrete_Event_Simulation-orange)
![PyGame](https://img.shields.io/badge/PyGame-Live_Visualizer-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data_Analysis-blueviolet)

This repository contains the code, data visualizations, and project report for a **Digital Twin** of the Radhanagari State Transport (ST) Depot in Maharashtra, India. 

Traditional transit planning often measures success by comparing *total daily seats* against *total daily passengers*. This project proves mathematically that this "average" approach hides catastrophic system failures. By modeling irrational human behavior against strict physical bus timetables, this simulation reveals severe spatiotemporal mismatches, ghost buses, and dangerous platform bottlenecks.

## ✨ Key Features

* **Discrete-Event Simulation Engine:** Built with Python's `SimPy` to handle strict time-step physics, bus capacities, and localized event loops over a 12-hour window.
* **Non-Homogeneous Passenger Generation:** Models actual urban tidal flows (morning and evening rushes) rather than flat arrival rates.
* **"Smart Commuter" Behavioral AI:** Passengers don't just wait passively. They use *Tappa-Tappa* (step-by-step) routing, actively seeking partial transfer routes if they get desperate.
* **Mob-Boarding Queue Dynamics:** Abandons standard FIFO logic. The station is modeled as a "Total System Backlog," simulating the chaotic rush when a bus arrives based on passenger urgency scores.
* **Live 1080p Visualizer:** A stunning `Pygame` dashboard that visualizes the simulation in real-time, showing queue buildups, dispatch logs, and rush-hour timelines.
* **Monte Carlo Stability:** Runs 30 back-to-back simulations to guarantee data convergence via the Central Limit Theorem.

## 📊 Major Findings

Through 30 simulated days, the Digital Twin uncovered the following realities at the depot:
1. **The 42.7% Paradox:** Despite adequate total daily seating, the weekday boarding success rate is only 42.7%, leaving an average of **760 passengers stranded daily**.
2. **Platform Bottlenecks:** The rigid bus schedule collides with the 5:00 PM synchronized release of schools and offices, trapping over 185+ people simultaneously in the station footprint.
3. **Ghost Buses & Route Collapse:** Local routes (Nipani, Kolhapur) slam into the 60-passenger physical limit during rush hours. Meanwhile, long-haul express routes (Panjim, Latur) depart during rush hours completely empty (burning fuel for 0 passengers).
4. **Schedule Starvation:** Uncovered structural flaws, such as the Mudaltitta-Gargoti route having zero scheduled evening departures, permanently stranding students.

**Conclusion:** The depot does not need more buses. It needs data-driven route reallocation. 

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/into-the-bus-verse.git
   cd into-the-bus-verse
   ```

2. **Install the required dependencies:**
   This project relies on standard data science libraries and Pygame.
   ```bash
   pip install simpy numpy pandas matplotlib pygame
   ```

---

## 🚀 Usage

The project is split into two executable components:

### 1. Run the Headless Simulation & Generate Plots
Run `simulation.py` to execute the 30-day Monte Carlo simulation in the terminal. It will calculate all queueing mathematics, print an executive summary, and generate 7 high-resolution data plots.
```bash
python simulation.py
```
*Output will be saved in a new folder named `radhanagari_plots/`.*

### 2. Launch the Live Presentation Dashboard
Run `demo.py` to open the 1080p Pygame live visualizer. This connects directly to the SimPy engine to show buses arriving, queues dynamically changing, and live KPI tracking.
```bash
python demo.py
```
*Note: Press `ESC` to exit the fullscreen dashboard.*

---

## 📁 Repository Structure

```text
├── simulation.py              # Core SimPy engine, AI logic, and Matplotlib charting
├── demo.py                    # Pygame 1080p Live Visualizer dashboard
├── Into_the_Bus_Verse_Report.pdf # Final academic project report (Methodology & Math)
├── Into_the_Bus_Verse.pptx    # Slide deck for presentation
├── radhanagari_plots/         # Directory containing the generated analytical graphs
│   ├── 01_Tidal_Flow.png
│   ├── 04_Route_Efficiency.png
│   ├── 05_Bottleneck.png
│   └── ...
└── README.md
```

## 👨‍💻 Author

**Rupesh Hemant Bhusare** <br>
Indian Institute of Technology (IIT) Goa  
Course: CS572 - Modeling and Simulation of Systems (2026)  

---
*If you find this project interesting for urban planning or operations research, feel free to fork, star, or reach out!*
