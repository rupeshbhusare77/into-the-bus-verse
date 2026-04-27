"""
Into the Bus-Verse: Modeling Passenger Behavior and Bus Dynamics
in a Multi-Route Transit System

A discrete-event simulation using Python + SimPy
"""

import os
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# ─────────────────────────────────────────────
#  CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────

RANDOM_SEED = 42
SIM_DURATION = 720          # 12 hours in minutes (6 AM to 6 PM)
NUM_RUNS = 30               # Number of runs for CLT convergence

# Route definitions based STRICTLY on the visual Color Map
ROUTES = {
    # Red Route (Direct North)
    "Kolhapur_Express": ["Radhanagari", "Rashivade", "Parite", "Kolhapur"], 
    # Purple -> Nipani
    "Nipani_Local": ["Radhanagari", "Mudal_Titta", "Nipani"], 
    # Purple -> Green
    "Mudaltitta_Gargoti": ["Radhanagari", "Mudal_Titta", "Gargoti"], 
    # Purple -> Pink (Extended)
    "Ichalkaranji_Sangli": ["Radhanagari", "Mudal_Titta", "Nipani", "Kagal", "Hupari", "Ichalkaranji", "Sangli"], 
    # Red -> North
    "Pune_Mumbai": ["Radhanagari", "Kolhapur", "Karad", "Pune", "Mumbai"], 
    # Yellow Exception (6:00 AM)
    "Pune_Mumbai_Via_Mudal": ["Radhanagari", "Mudal_Titta", "Kolhapur", "Karad", "Pune", "Mumbai"],
    # Black Route (West)
    "Panjim_Express": ["Radhanagari", "Sawantwadi", "Panjim"],
    # Red -> East
    "Latur_Express": ["Radhanagari", "Kolhapur", "Sangli", "Pandharpur", "Latur"]
}

TRANSFER_HUBS = ["Kolhapur", "Mudal_Titta", "Nipani"]

# Deterministic Timetable
BUS_TIMETABLES = {
    "Kolhapur_Express": [(0, "through"), (30, "through"), (60, "origin"), (90, "through"), 
                         (120, "through"), (150, "origin"), (210, "through"), (300, "origin"), 
                         (420, "through"), (510, "through"), (620, "origin"), (690, "through")],
    
    "Nipani_Local": [(0, "origin"), (60, "through"), (120, "origin"), (150, "through"), 
                     (360, "origin"), (420, "through"), (600, "origin"), (675, "through")],
    
    "Mudaltitta_Gargoti": [(90, "origin"), (225, "through"), (300, "origin"), (330, "through")],
    
    "Ichalkaranji_Sangli": [(360, "origin"), (390, "through"), (405, "origin"), (450, "through"), (570, "origin")],
    
    "Pune_Mumbai": [(120, "origin"), (180, "through"), (480, "through")],
    
    "Pune_Mumbai_Via_Mudal": [(0, "through")], 
    
    "Panjim_Express": [(40, "through"), (165, "origin"), (330, "through"), (465, "through"), (525, "through")],
    
    "Latur_Express": [(165, "origin"), (525, "through")]
}

BUS_SEAT_CAPACITY = 40
BUS_STANDING_CAPACITY = 20   
BOARDING_TIME = 0.05  # 3 seconds bottleneck per person to board

PASSENGER_TYPE_PROBS_WEEKDAY = {"student": 0.40, "worker": 0.35, "casual": 0.25}
PASSENGER_TYPE_PROBS_WEEKEND = {"student": 0.05, "worker": 0.15, "casual": 0.80}

# Rush hour periods (minutes from sim start at 6:00 AM)
MORNING_RUSH = (120, 240)   # 8:00 AM to 10:00 AM
MIDDAY_LOW   = (240, 600)   # 10:00 AM to 4:00 PM
EVENING_RUSH = (600, 720)   # 4:00 PM to 6:00 PM

# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class PassengerRecord:
    pid: int
    ptype: str
    arrival_time: float
    destination: str
    preferred_route: str
    urgency: float
    comfort_pref: float
    patience: float         
    boarding_time: Optional[float] = None
    boarded: bool = False
    left_without_boarding: bool = False
    boarded_route: Optional[str] = None
    waiting_time: Optional[float] = None
    is_transfer: bool = False

@dataclass
class BusRecord:
    bus_id: str
    route: str
    scheduled_time: float
    actual_arrival: float
    departure_time: Optional[float] = None
    passengers_boarded: int = 0
    passengers_onboard_arrival: int = 0
    load_at_departure: int = 0

_current_run_passengers: List[PassengerRecord] = []
_current_run_buses: List[BusRecord] = []
_current_run_queue_log: List[Tuple[float, int]] = []

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def arrival_rate_at(t: float, is_weekend: bool) -> float:
    """Returns the expected number of passengers arriving per minute.
    Tuned for a Busy Regional Depot (1,000 - 1,500 passengers/day)."""
    if is_weekend:
        # Weekend peaks are softer, midday is quieter
        if MORNING_RUSH[0] <= t < MORNING_RUSH[1] or EVENING_RUSH[0] <= t < EVENING_RUSH[1]:
            return 1.5 
        else:
            return 0.8
    else:
        # Weekday tuning to hit ~1,400 total daily passengers
        if MORNING_RUSH[0] <= t < MORNING_RUSH[1]: 
            return 2.0  # Morning surge
        elif MIDDAY_LOW[0] <= t < MIDDAY_LOW[1]: 
            return 0.5  # Dead midday hours
        elif EVENING_RUSH[0] <= t < EVENING_RUSH[1]: 
            return 1.6  # Evening surge
        else:
            return 0.5  # Early morning / late evening off-peak

def get_passenger_type(is_weekend: bool, rng: np.random.Generator) -> str:
    """Determines if the passenger is a student, worker, or casual traveler."""
    probs = PASSENGER_TYPE_PROBS_WEEKEND if is_weekend else PASSENGER_TYPE_PROBS_WEEKDAY
    return rng.choice(list(probs.keys()), p=list(probs.values()))

def get_passenger_params(ptype: str, rng: np.random.Generator) -> Tuple[float, float, float, str, str]:
    """Assigns realistic Patience limits and Destinations based on demographic."""
    if ptype == "student":
        urgency = rng.uniform(0.6, 1.0)
        comfort = rng.uniform(0.0, 0.4)
        patience = rng.normal(60, 15)  # Willing to wait ~1 hour
        possible_dests = ["Kolhapur", "Nipani", "Mudal_Titta", "Gargoti"]
        
    elif ptype == "worker":
        urgency = rng.uniform(0.7, 1.0)
        comfort = rng.uniform(0.2, 0.6)
        patience = rng.normal(45, 15)  # Willing to wait ~45 minutes
        possible_dests = ["Kolhapur", "Nipani", "Mudal_Titta", "Gargoti", "Ichalkaranji"]
        
    else: # Casual
        urgency = rng.uniform(0.1, 0.6)
        comfort = rng.uniform(0.4, 1.0)
        patience = rng.normal(180, 45) # Taking long haul, willing to wait ~3 hours
        possible_dests = []
        for route in ROUTES.values():
            for stop in route:
                if stop != "Radhanagari":
                    possible_dests.append(stop)
        possible_dests = list(set(possible_dests))

    destination = rng.choice(possible_dests)
    viable_routes = [r for r, stops in ROUTES.items() if destination in stops]
    preferred_route = rng.choice(viable_routes)
    
    return urgency, comfort, float(max(patience, 15.0)), destination, preferred_route

# ─────────────────────────────────────────────
#  SIMPY PROCESSES (CORE AI & PHYSICS)
# ─────────────────────────────────────────────

class BusStation:
    def __init__(self, env: simpy.Environment, is_weekend: bool, rng: np.random.Generator):
        self.env = env
        self.is_weekend = is_weekend
        self.rng = rng
        
        self.queues: Dict[str, List[PassengerRecord]] = {r: [] for r in ROUTES}
        self.bus_arrival_events: Dict[str, simpy.Event] = {r: env.event() for r in ROUTES}
        
        self._pid = 0
        self._bid = 0
        
        self.env.process(self.queue_logger())

    def _new_pid(self): 
        self._pid += 1
        return self._pid
        
    def _new_bid(self): 
        self._bid += 1
        return self._bid

    def queue_logger(self):
        """Logs the true physical queue size every 5 minutes for plotting."""
        while True:
            total_waiting = sum(len(q) for q in self.queues.values())
            _current_run_queue_log.append((self.env.now, total_waiting))
            yield self.env.timeout(5)

    def _get_next_bus_wait(self, route: str) -> float:
        """Looks at the timetable to see how far away the next bus is."""
        schedule = BUS_TIMETABLES[route]
        for t, _ in schedule:
            if t > self.env.now: 
                return t - self.env.now
        return 999.0 # No more buses left today

    def passenger_process(self, record: PassengerRecord):
        """Smart Pathfinding & Transfer Logic."""
        _current_run_passengers.append(record)

        # 1. Find direct routes
        direct_routes = []
        for r, stops in ROUTES.items():
            if record.destination in stops:
                direct_routes.append(r)
                
        # 2. Find partial routes (transfers) if passenger is urgent
        partial_routes = []
        if record.urgency > 0.60:  
            for r, stops in ROUTES.items():
                if r not in direct_routes:
                    for hub in TRANSFER_HUBS:
                        if hub in stops:
                            partial_routes.append(r)
                            break

        # 3. Combine choices
        active_routes = direct_routes + partial_routes
        if len(active_routes) == 0:
            active_routes = [record.preferred_route]

        # 4. Join the physical queue
        if len(direct_routes) > 0:
            primary_route = direct_routes[0]
        else:
            primary_route = active_routes[0]
            
        self.queues[primary_route].append(record)

        deadline = self.env.now + record.patience
        boarded = False

        # 5. Wait for ANY viable bus to arrive
        while self.env.now < deadline and not boarded:
            events = [self.bus_arrival_events[r] for r in set(active_routes)]
            remaining_patience = deadline - self.env.now
            
            yield simpy.events.AnyOf(self.env, events) | self.env.timeout(remaining_patience)

            if self.env.now >= deadline: 
                break 
                
            yield self.env.timeout(0.01) # Allow boarding logic to execute
            
            if record.boarded:
                boarded = True
                # Log if they took a smart transfer route
                if record.boarded_route in partial_routes and record.boarded_route not in direct_routes:
                    record.is_transfer = True
                break

        # 6. Check if they gave up and left
        if not boarded:
            record.left_without_boarding = True
            record.waiting_time = self.env.now - record.arrival_time
            if record in self.queues[primary_route]: 
                self.queues[primary_route].remove(record)

    def _boarding_decision(self, p: PassengerRecord, seats_avail: bool, standing_avail: bool, load: int, route: str) -> bool:
        """Calculates boarding probability based on psychology and the timetable."""
        if not standing_avail: 
            return False 
            
        wait_for_next = self._get_next_bus_wait(route)
        actual_urgency = p.urgency
        
        # Desperation Multiplier
        if wait_for_next > 60:
            actual_urgency = min(1.0, actual_urgency + 0.5)
        elif wait_for_next > 30:
            actual_urgency = min(1.0, actual_urgency + 0.2)
            
        # 5% baseline hesitation in a crowded mob
        if self.rng.random() < 0.05: 
            return False 

        # Comfort seekers
        if not seats_avail and p.comfort_pref > 0.7 and actual_urgency < 0.6: 
            return False
            
        # Heavy crowding logic
        ratio = load / (BUS_SEAT_CAPACITY + BUS_STANDING_CAPACITY)
        if ratio > 0.85: 
            return self.rng.random() < (actual_urgency * (1 - p.comfort_pref * 0.5))
            
        if ratio > 0.6: 
            return self.rng.random() < (0.5 + actual_urgency * 0.4)
            
        return True

    def bus_process(self, route: str, scheduled_time: float, bus_type: str):
        """Handles Arrival, Early/Late Delays, Tidal Flow, and the Chaotic Mob."""
        env = self.env
        rng = self.rng
        
        # --- EARLY/LATE PHYSICS ---
        delay = rng.normal(0, 10) 
        actual_arrival = max(0, scheduled_time + delay)
        yield env.timeout(actual_arrival)

        bus_id = f"{route}_Bus{self._new_bid()}"
        
        if bus_type == "through":
            onboard = int(rng.uniform(15, 35))
        else:
            onboard = 0
            
        record = BusRecord(bus_id, route, scheduled_time, env.now, passengers_onboard_arrival=onboard)
        _current_run_buses.append(record)

        # --- TIDAL FLOW (Alighting Passengers) ---
        if onboard > 0:
            alighting = int(rng.uniform(0, onboard * 0.6))
            if alighting > 0:
                yield env.timeout(1.0) # Time to step off the bus
                onboard -= alighting
                
                # Time of Day logic for Radhanagari
                if env.now < 300: 
                    transfer_ratio = 0.70 # Morning
                elif env.now < 480: 
                    transfer_ratio = 0.50 # Midday
                else: 
                    transfer_ratio = 0.30 # Evening
                    
                transfer_count = int(alighting * transfer_ratio)
                for _ in range(transfer_count): 
                    self.env.process(self._handle_transfer())

        # --- DEPARTURE DELAY LOGIC ---
        if actual_arrival < scheduled_time:
            # Bus is early, must wait until scheduled time + 5 min loading window
            wait_end = scheduled_time + 5 
        else:
            # Bus is late, rushed 5 min cooldown window
            wait_end = actual_arrival + 5 

        # Wake up waiting passengers
        self.bus_arrival_events[route].succeed()
        self.bus_arrival_events[route] = env.event()

        # --- THE CHAOTIC MOB MECHANIC ---
        # Queue is completely scrambled based on urgency + random luck
        self.queues[route].sort(key=lambda p: p.urgency + rng.uniform(-0.2, 0.2), reverse=True)

        tot_cap = BUS_SEAT_CAPACITY + BUS_STANDING_CAPACITY
        
        while env.now < wait_end and onboard < tot_cap:
            if len(self.queues[route]) == 0: 
                yield env.timeout(1.0)
                continue
            
            p = self.queues[route][0]
            seats_avail = onboard < BUS_SEAT_CAPACITY
            
            if self._boarding_decision(p, seats_avail, True, onboard, route):
                self.queues[route].pop(0)
                p.boarded = True
                p.boarding_time = env.now
                p.boarded_route = route
                p.waiting_time = env.now - p.arrival_time
                onboard += 1
                record.passengers_boarded += 1
                yield env.timeout(BOARDING_TIME) # 3 seconds per person physical bottleneck
            else:
                # Passenger refuses to board, steps out of the doorway
                self.queues[route].append(self.queues[route].pop(0)) 
                yield env.timeout(0.1) # 6 seconds delay

        record.departure_time = env.now
        record.load_at_departure = onboard

    def _handle_transfer(self):
        """Generates a passenger who just got off a through-bus and needs a new route."""
        yield self.env.timeout(self.rng.uniform(1.0, 3.0)) # Walking across the station
        
        ptype = get_passenger_type(self.is_weekend, self.rng)
        urg, com, pat, dest, pref = get_passenger_params(ptype, self.rng)
        
        # High urgency because they are transferring mid-journey
        record = PassengerRecord(
            self._new_pid(), ptype, self.env.now, dest, pref, 0.9, com, pat, is_transfer=True
        )
        self.env.process(self.passenger_process(record))

    def run_generator(self):
        """Spawns passengers over time based on the tidal flow rates."""
        while True:
            rate = arrival_rate_at(self.env.now, self.is_weekend)
            inter_arrival_time = self.rng.exponential(1.0 / rate)
            yield self.env.timeout(inter_arrival_time)
            
            ptype = get_passenger_type(self.is_weekend, self.rng)
            
            # Group clusters
            cluster_size = 1
            if not self.is_weekend:
                if ptype == "student":
                    cluster_size = self.rng.integers(1, 5)
                elif ptype == "worker":
                    cluster_size = self.rng.integers(1, 3)

            for _ in range(cluster_size):
                urg, com, pat, dest, pref = get_passenger_params(ptype, self.rng)
                record = PassengerRecord(
                    self._new_pid(), ptype, self.env.now, dest, pref, urg, com, pat
                )
                self.env.process(self.passenger_process(record))

    def run_scheduler(self):
        """Initializes all buses for the day based on the timetable."""
        for route, sched in BUS_TIMETABLES.items():
            for t, btype in sched: 
                self.env.process(self.bus_process(route, t, btype))

# ─────────────────────────────────────────────
#  EXECUTION & METRICS
# ─────────────────────────────────────────────

def run_single(seed: int, is_weekend: bool):
    """Executes exactly one 12-hour day of simulation."""
    global _current_run_passengers, _current_run_buses, _current_run_queue_log
    _current_run_passengers = []
    _current_run_buses = []
    _current_run_queue_log = []
    
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    
    station = BusStation(env, is_weekend, rng)
    env.process(station.run_generator())
    station.run_scheduler()
    
    env.run(until=SIM_DURATION)
    
    return list(_current_run_passengers), list(_current_run_buses), list(_current_run_queue_log)

def compute_metrics(p_list, b_list):
    """Calculates all statistics for a single simulation run."""
    df_p = pd.DataFrame([vars(p) for p in p_list])
    df_b = pd.DataFrame([vars(b) for b in b_list])
    
    b_df = df_p[df_p["boarded"] == True]
    
    metrics = {
        "total": len(df_p), 
        "boarded": len(b_df), 
        "stranded": len(df_p[df_p["left_without_boarding"] == True]),
        "rate": len(b_df) / max(len(df_p), 1), 
        "wait": b_df["waiting_time"].mean() if len(b_df) > 0 else 0
    }
    
    for pt in ["student", "worker", "casual"]:
        s = df_p[(df_p["ptype"] == pt) & (df_p["boarded"] == True)]
        metrics[f"wait_{pt}"] = s["waiting_time"].mean() if len(s) > 0 else 0
        
        total_type = len(df_p[df_p["ptype"] == pt])
        metrics[f"rate_{pt}"] = len(s) / max(total_type, 1)
    
    if len(df_b) > 0:
        metrics["occ"] = (df_b["load_at_departure"] / 60).mean()
        metrics["late_buses"] = len(df_b[df_b["actual_arrival"] > df_b["scheduled_time"] + 2]) / max(len(df_b), 1)
        
        for r in ROUTES: 
            route_buses = df_b[df_b["route"] == r]
            if len(route_buses) > 0:
                metrics[f"occ_{r}"] = (route_buses["load_at_departure"] / 60).mean()
            else:
                metrics[f"occ_{r}"] = 0
    else:
        metrics["occ"] = 0
        metrics["late_buses"] = 0
        
    return metrics

def run_experiment():
    """Runs the simulation NUM_RUNS times to satisfy the Central Limit Theorem."""
    res = {
        "wd": {"p": [], "b": [], "q": [], "m": []}, 
        "we": {"p": [], "b": [], "q": [], "m": []}
    }
    
    for i in range(NUM_RUNS):
        # Weekday Run
        p, b, q = run_single(RANDOM_SEED + i, False)
        res["wd"]["p"].append(p)
        res["wd"]["b"].append(b)
        res["wd"]["q"].append(q)
        res["wd"]["m"].append(compute_metrics(p, b))
        
        # Weekend Run
        p, b, q = run_single(RANDOM_SEED + i + 1000, True)
        res["we"]["p"].append(p)
        res["we"]["b"].append(b)
        res["we"]["q"].append(q)
        res["we"]["m"].append(compute_metrics(p, b))
        
        if (i + 1) % 5 == 0: 
            print(f"  [>] Runs complete: {i + 1}/{NUM_RUNS}")
            
    return res

# ─────────────────────────────────────────────
#  PLOTTING ENGINE
# ─────────────────────────────────────────────

def make_plots(res, save_dir="radhanagari_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Tidal Flow (Averaged per day)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("The Tidal Flow: Average Daily Arrivals", fontsize=14, fontweight="bold")
    
    scenarios = [("wd", "Weekday"), ("we", "Weekend")]
    for ax, (scen, title) in zip(axes, scenarios):
        arrs = [p.arrival_time for pl in res[scen]["p"] for p in pl]
        weights = np.ones_like(arrs) / NUM_RUNS # Calculates the per-day average mathematically
        
        color = "#2b5b84" if scen == "wd" else "#c06c84"
        ax.hist(arrs, bins=np.arange(0, 750, 30), weights=weights, color=color, edgecolor="white")
        
        ax.axvspan(120, 240, alpha=0.1, color="red", label="Morning Rush")
        ax.axvspan(600, 720, alpha=0.1, color="orange", label="Evening Rush")
        ax.set_title(title)
        ax.set_xlabel("Minutes from 6AM")
        ax.set_ylabel("Avg Passengers")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_Tidal_Flow.png", dpi=200)
    plt.close()

    # Plot 2: Wait Times (Survival of Fittest)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Survival of the Fittest: Average Wait Times (Boarded)", fontsize=14, fontweight="bold")
    
    for ax, (scen, title) in zip(axes, scenarios):
        m = pd.DataFrame(res[scen]["m"]).mean()
        categories = ["Student", "Worker", "Casual"]
        values = [m["wait_student"], m["wait_worker"], m["wait_casual"]]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        
        bars = ax.bar(categories, values, color=colors, edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel("Minutes")
        
        for b in bars: 
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, 
                    f"{b.get_height():.1f}m", ha="center", fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(f"{save_dir}/02_Wait_Times.png", dpi=200)
    plt.close()

    # Plot 3: Weekday vs Weekend Overall Stats
    m_wd = pd.DataFrame(res["wd"]["m"]).mean()
    m_we = pd.DataFrame(res["we"]["m"]).mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    x = np.arange(3)
    
    wd_stats = [m_wd["rate"] * 100, m_wd["occ"] * 100, m_wd["late_buses"] * 100]
    we_stats = [m_we["rate"] * 100, m_we["occ"] * 100, m_we["late_buses"] * 100]
    
    ax.bar(x - w/2, wd_stats, w, label="Weekday", color="#4C72B0")
    ax.bar(x + w/2, we_stats, w, label="Weekend", color="#DD8452")
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Boarding Success (%)", "Bus Occupancy (%)", "Buses Arriving Late (%)"])
    ax.set_title("System Diagnostics: Weekday vs Weekend", fontweight="bold")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/03_System_Diagnostics.png", dpi=200)
    plt.close()

    # Plot 4: Route Efficiency
    fig, ax = plt.subplots(figsize=(12, 6))
    keys = list(ROUTES.keys())
    vals = [m_wd[f"occ_{k}"] * 100 for k in keys]
    
    labels = [k.replace("_", " ") for k in keys]
    bars = ax.barh(labels, vals, color="#34495e", edgecolor="black")
    
    ax.axvline(100, color="red", linestyle="--", label="Full Capacity")
    ax.set_title("Route Efficiency: Avg Bus Occupancy (Weekday)", fontweight="bold")
    ax.set_xlabel("Occupancy Percentage (%)")
    ax.legend()
    
    for b, v in zip(bars, vals): 
        ax.text(v + 1, b.get_y() + b.get_height()/2, f"{v:.1f}%", va="center")
        
    plt.tight_layout()
    plt.savefig(f"{save_dir}/04_Route_Efficiency.png", dpi=200)
    plt.close()

    # Plot 5: True Queue Bottleneck (Run 1)
    q_data = res["wd"]["q"][0]
    t, l = zip(*q_data)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(t, l, alpha=0.4, color="#e74c3c")
    ax.plot(t, l, color="#c0392b", lw=2)
    
    ax.set_title("Station Crowd Density: True Platform Bottleneck (Single Day)", fontweight="bold")
    ax.set_xlabel("Minutes from 6AM")
    ax.set_ylabel("People Waiting")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/05_Bottleneck.png", dpi=200)
    plt.close()

    # Plot 6: Central Limit Theorem (Mathematical Proof)
    fig, ax = plt.subplots(figsize=(10, 4))
    run_waits = [m["wait"] for m in res["wd"]["m"]]
    cum_means = [np.mean(run_waits[:i+1]) for i in range(len(run_waits))]
    
    ax.plot(range(1, 31), cum_means, marker="o", color="#8e44ad")
    ax.axhline(cum_means[-1], color="grey", linestyle="--", label=f"Converged Mean: {cum_means[-1]:.2f} min")
    
    ax.set_title("Central Limit Theorem: Wait Time Convergence", fontweight="bold")
    ax.set_xlabel("Simulation Runs")
    ax.set_ylabel("Cumulative Mean Wait (min)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/06_CLT_Proof.png", dpi=200)
    plt.close()

    # Plot 7: Bus Occupancy Timeline by Route (Weekday, Single Run)
    # We use Run 0 to show a realistic, discrete single day rather than a smeared average
    b_data = res["wd"]["b"][0] 
    df_b = pd.DataFrame([vars(b) for b in b_data])
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle("Bus Occupancy Over Time by Route (Weekday)", fontsize=16, fontweight="bold")
    
    axes = axes.flatten()
    for i, route in enumerate(ROUTES.keys()):
        ax = axes[i]
        
        # Filter for the specific route and sort by when the bus actually left
        route_buses = df_b[df_b["route"] == route].sort_values("departure_time")
        
        if not route_buses.empty:
            times = route_buses["departure_time"]
            loads = route_buses["load_at_departure"]
            
            # Draw the line and fill under it
            ax.plot(times, loads, marker='o', markersize=6, linestyle='-', color="#2c3e50", linewidth=2)
            ax.fill_between(times, loads, alpha=0.3, color="#34495e")
            
            # Add data labels slightly above each dot
            for x, y in zip(times, loads):
                ax.text(x, y + 2, str(int(y)), ha='center', va='bottom', fontsize=9)
                
        # Draw the physical capacity limit line
        ax.axhline(60, color="#e74c3c", linestyle="--", alpha=0.8, linewidth=1.5)
        
        ax.set_title(route.replace("_", " "), fontweight="bold")
        ax.set_ylim(0, 75) # Extra space above 60 for the data labels
        
        # Only put X-axis labels on the bottom row to keep it clean
        if i >= 6: 
            ax.set_xlabel("Minutes from 6:00 AM")
        if i % 2 == 0: 
            ax.set_ylabel("Passengers Onboard")
            
    plt.tight_layout()
    plt.savefig(f"{save_dir}/07_Occupancy_Timeline.png", dpi=200)
    plt.close()

# ─────────────────────────────────────────────
#  MAIN EXECUTION BLOCK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*65)
    print(" 🚌 RADHANAGARI ST DEPOT: DIGITAL TWIN (V2.0 STABLE)")
    print("="*65)
    
    res = run_experiment()
    
    print("\n" + "="*65)
    print(" 📊 EXECUTIVE SUMMARY REPORT")
    print("="*65)
    
    for s, title in [("wd", "WEEKDAY"), ("we", "WEEKEND")]:
        m = pd.DataFrame(res[s]["m"]).mean()
        print(f"\n >>> {title} AVERAGES <<<")
        print(f" Daily Foot Traffic : {m['total']:.0f} passengers")
        print(f" Boarding Success   : {m['rate']*100:.1f}% (Stranded/Left: {m['stranded']:.0f})")
        print(f" System Bus Load    : {m['occ']*100:.1f}% Capacity")
        print(f" Buses Arriving Late: {m['late_buses']*100:.1f}%")
        print(f" Demographics:")
        print(f"  - Students: {m['wait_student']:.1f}m wait | {m['rate_student']*100:.1f}% Boarded")
        print(f"  - Workers:  {m['wait_worker']:.1f}m wait | {m['rate_worker']*100:.1f}% Boarded")
        print(f"  - Casuals:  {m['wait_casual']:.1f}m wait | {m['rate_casual']*100:.1f}% Boarded")
    
    print("\n[!] Generating all 6 data plots...")
    make_plots(res)
    print("[✓] Success! Check 'radhanagari_plots/' for finalized charts.")
    print("="*65)