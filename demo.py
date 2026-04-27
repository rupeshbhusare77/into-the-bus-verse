"""
Radhanagari Transit Authority — Full HD (1080p) Presentation Visualizer
Features fixed-axis live graphing, rush-hour timelines, and sharp DPI rendering.
"""

import threading
import sys
import os
import ctypes
import collections
import simpy
import simpy.rt
import numpy as np
import pygame



# --- 1. CRISP 1080p OVERRIDE ---
# Tells Windows to stop artificially zooming/blurring the window
try:
    if os.name == 'nt':
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

try:
    import simulation
except ImportError:
    print("ERROR: live_visualizer.py must be in the same directory as simulation.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG & PALETTE
# ─────────────────────────────────────────────────────────────────────────────

SPEED_FACTOR    = 0.08  
FPS             = 60
WIN_W, WIN_H    = 1920, 1080  

C = {
    "bg":           (18, 18, 20),
    "panel":        (28, 28, 32),
    "border":       (50, 50, 55),
    "accent":       (88, 166, 255),    # Blue
    "accent_warn":  (255, 166, 87),    # Amber
    "accent_err":   (248, 81, 73),     # Red
    "accent_ok":    (63, 185, 80),     # Green
    "text_main":    (230, 237, 243),
    "text_dim":     (139, 148, 158),
    "bus":          (248, 81, 73),
    "graph_fill":   (88, 166, 255, 40), # Transparent blue
    "student":      (137, 180, 250),   # Professional Blue
    "worker":       (249, 226, 175),   # Professional Amber
    "casual":       (166, 227, 161),   # Professional Green
}

# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATION THREAD & GRAPHING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(station, env):
    simulation._current_run_passengers = []
    simulation._current_run_buses      = []
    env.process(station.run_generator())
    station.run_scheduler()
    env.run(until=simulation.SIM_DURATION)

class LiveGraph:
    def __init__(self, max_points=720):
        # 720 points = 1 point per minute for 12 hours
        self.data = collections.deque(maxlen=max_points)
        self.last_update_time = -1

    def update(self, current_time, value):
        if current_time - self.last_update_time >= 1.0:
            self.data.append(value)
            self.last_update_time = current_time

    def draw(self, surf, rect, color, font):
        x, y, w, h = rect
        if len(self.data) < 2:
            return

        max_val = max(max(self.data), 10) 
        
        points = []
        for i, val in enumerate(self.data):
            # FIXED: We scale the X position by maxlen (720) so the graph draws left-to-right!
            px = x + int(i / (self.data.maxlen - 1) * w)
            py = y + h - int((val / max_val) * h)
            points.append((px, py))

        fill_points = [(x, y + h)] + points + [(points[-1][0], y + h)]
        fill_surface = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        pygame.draw.polygon(fill_surface, C["graph_fill"], fill_points)
        surf.blit(fill_surface, (0, 0))

        pygame.draw.lines(surf, color, False, points, 3)

        img = font.render(str(max_val), True, C["text_dim"])
        surf.blit(img, (x - 30, y))

# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_rect(surf, color, rect, radius=8, border=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border, rect, width=1, border_radius=radius)

def draw_text(surf, text, pos, font, color=C["text_main"], anchor="topleft"):
    img = font.render(str(text), True, color)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

class PresentationDashboard:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Radhanagari Transit - Live Simulation")
        
        # FIXED: Fullscreen mode with un-scaled pixels for maximum sharpness
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.FULLSCREEN)
        self.clock  = pygame.time.Clock()

        font_family = "segoeui" if sys.platform == "win32" else "arial"
        self.fonts = {
            "title": pygame.font.SysFont(font_family, 32, bold=True),
            "h1":    pygame.font.SysFont(font_family, 20, bold=True),
            "h2":    pygame.font.SysFont(font_family, 18, bold=True),
            "body":  pygame.font.SysFont(font_family, 18),
            "big":   pygame.font.SysFont(font_family, 42, bold=True),
            "small": pygame.font.SysFont(font_family, 14)
        }

        self.env     = simpy.rt.RealtimeEnvironment(factor=SPEED_FACTOR, strict=False)
        self.rng     = np.random.default_rng(simulation.RANDOM_SEED)
        self.station = simulation.BusStation(self.env, is_weekend=False, rng=self.rng)
        
        self.route_names = list(simulation.ROUTES.keys())
        self.queue_graph = LiveGraph(max_points=720) 

        self.sim_thread = threading.Thread(target=run_simulation, args=(self.station, self.env), daemon=True)
        self.sim_thread.start()

    def get_sim_time(self):
        raw_min = int(self.env.now)
        hour = (raw_min // 60) + 6
        minute = raw_min % 60
        am_pm = "AM" if hour < 12 else "PM"
        display_hour = hour if hour <= 12 else hour - 12
        return display_hour, minute, am_pm, raw_min

    # ─────────────────────────────────────────────────────────────────────────
    #  COMPONENTS
    # ─────────────────────────────────────────────────────────────────────────

    def draw_header(self):
        draw_rect(self.screen, C["panel"], (0, 0, WIN_W, 80))
        pygame.draw.line(self.screen, C["border"], (0, 80), (WIN_W, 80), 2)

        # Main Title
        draw_text(self.screen, "RADHANAGARI TRANSIT SIMULATION", (30, 24), self.fonts["title"], C["accent"])
        
        # --- NEW: Demographics Legend ---
        legend_x = WIN_W - 550
        
        # Student Legend
        pygame.draw.circle(self.screen, C["student"], (legend_x, 40), 7)
        draw_text(self.screen, "Student", (legend_x + 15, 28), self.fonts["body"], C["text_main"])
        
        # Worker Legend
        pygame.draw.circle(self.screen, C["worker"], (legend_x + 100, 40), 7)
        draw_text(self.screen, "Worker", (legend_x + 115, 28), self.fonts["body"], C["text_main"])
        
        # Casual Legend
        pygame.draw.circle(self.screen, C["casual"], (legend_x + 200, 40), 7)
        draw_text(self.screen, "Casual", (legend_x + 215, 28), self.fonts["body"], C["text_main"])
        # --------------------------------

        # Clock - Centered
        h, m, ap, _ = self.get_sim_time()
        draw_text(self.screen, f"SYSTEM TIME: {h:02d}:{m:02d} {ap}", (WIN_W // 2, 40), self.fonts["title"], C["text_main"], "midtop")

    def draw_kpi_row(self, y_offset):
        buses = simulation._current_run_buses
        passengers = simulation._current_run_passengers

        boarded = sum(1 for p in passengers if p.boarded)
        stranded = sum(1 for p in passengers if p.left_without_boarding)
        waiting = sum(len(q) for q in self.station.queues.values())
        active_buses = sum(1 for b in buses if b.actual_arrival <= self.env.now and b.departure_time is None)

        kpis = [
            ("TOTAL PASSENGERS", len(passengers), C["accent"]),
            ("SUCCESSFULLY BOARDED", boarded, C["accent_ok"]),
            ("LEFT STRANDED", stranded, C["accent_err"]),
            ("CURRENTLY WAITING", waiting, C["accent_warn"]),
            ("ACTIVE BUSES AT DEPOT", active_buses, C["text_main"])
        ]

        card_w = (WIN_W - 60 - (len(kpis) - 1) * 20) // len(kpis)
        
        for i, (label, val, color) in enumerate(kpis):
            cx = 30 + i * (card_w + 20)
            draw_rect(self.screen, C["panel"], (cx, y_offset, card_w, 110), border=C["border"])
            draw_text(self.screen, label, (cx + 20, y_offset + 20), self.fonts["small"], C["text_dim"])
            draw_text(self.screen, str(val), (cx + 20, y_offset + 45), self.fonts["big"], color)

    def draw_timeline(self, rect):
        x, y, w, h = rect
        draw_rect(self.screen, C["panel"], rect, border=C["border"])
        
        draw_text(self.screen, "SCHEDULE TIMELINE", (x + 30, y + h//2), self.fonts["h2"], C["text_dim"], "midleft")

        # Map minutes (0-720) to pixels
        line_start = x + 250
        line_w = w - 280
        def time_to_x(t): return line_start + int((t / 720) * line_w)

        # Draw transparent Rush Hour zones
        zone_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        # Morning (120 - 240 mins) = Red
        pygame.draw.rect(zone_surf, (*C["accent_err"], 50), (time_to_x(120), y + 15, time_to_x(240) - time_to_x(120), h - 30), border_radius=4)
        # Evening (600 - 720 mins) = Amber
        pygame.draw.rect(zone_surf, (*C["accent_warn"], 50), (time_to_x(600), y + 15, time_to_x(720) - time_to_x(600), h - 30), border_radius=4)
        self.screen.blit(zone_surf, (0, 0))

        # Base Line
        pygame.draw.line(self.screen, C["border"], (time_to_x(0), y + h//2), (time_to_x(720), y + h//2), 4)

        # Time markers every 2 hours
        for hour in range(0, 13, 2):
            tx = time_to_x(hour * 60)
            pygame.draw.circle(self.screen, C["text_dim"], (tx, y + h//2), 6)
            lbl = f"{hour+6}AM" if hour+6 < 12 else (f"12PM" if hour+6==12 else f"{hour-6}PM")
            draw_text(self.screen, lbl, (tx, y + h//2 + 15), self.fonts["small"], C["text_dim"], "midtop")

        # Live Marker
        cx = time_to_x(self.env.now)
        pygame.draw.line(self.screen, C["accent"], (cx, y + 10), (cx, y + h - 10), 3)
        pygame.draw.polygon(self.screen, C["accent"], [(cx - 8, y + 10), (cx + 8, y + 10), (cx, y + 20)])
        
        # Rush Hour Labels
        draw_text(self.screen, "MORNING SURGE", (time_to_x(180), y + 25), self.fonts["small"], C["accent_err"], "midtop")
        draw_text(self.screen, "EVENING SURGE", (time_to_x(660), y + 25), self.fonts["small"], C["accent_warn"], "midtop")


    def draw_routes_panel(self, rect):
        x, y, w, h = rect
        draw_rect(self.screen, C["panel"], rect, border=C["border"])
        draw_text(self.screen, "LIVE TERMINAL PLATFORMS", (x + 30, y + 20), self.fonts["h1"], C["text_main"])
        pygame.draw.line(self.screen, C["border"], (x, y + 60), (x + w, y + 60), 2)

        active_map = {}
        for b in simulation._current_run_buses:
            if b.actual_arrival <= self.env.now and b.departure_time is None:
                active_map.setdefault(b.route, []).append(b)

        row_y = y + 75
        row_h = 75

        for route_name in self.route_names:
            clean_name = route_name.replace("_", " ")
            queue = self.station.queues.get(route_name, [])
            buses_here = active_map.get(route_name, [])

            draw_text(self.screen, clean_name, (x + 30, row_y + 15), self.fonts["h2"], C["accent"])
            draw_text(self.screen, f"Waiting: {len(queue)}", (x + 280, row_y + 15), self.fonts["body"], C["text_dim"])

            dot_x = x + 400
            max_dots = 30
            for i, pax in enumerate(queue):
                if i >= max_dots:
                    draw_text(self.screen, f"+{len(queue) - max_dots}", (dot_x + 10, row_y + 10), self.fonts["body"], C["text_dim"])
                    break
                # Default to white if patient type color isn't mapped
                color = C.get(pax.ptype, (255,255,255)) 
                pygame.draw.circle(self.screen, color, (dot_x, row_y + 24), 7)
                dot_x += 20

            bus_x = x + w - 260
            if buses_here:
                bus = buses_here[0]
                load = bus.passengers_boarded + bus.passengers_onboard_arrival
                cap = simulation.BUS_SEAT_CAPACITY + simulation.BUS_STANDING_CAPACITY
                
                draw_rect(self.screen, C["bus"], (bus_x, row_y, 220, 45), radius=6)
                draw_text(self.screen, f"BOARDING: {load}/{cap}", (bus_x + 15, row_y + 10), self.fonts["h2"], (255,255,255))
            else:
                draw_rect(self.screen, C["border"], (bus_x, row_y, 220, 45), radius=6)
                draw_text(self.screen, "STANDBY", (bus_x + 70, row_y + 10), self.fonts["h2"], C["text_dim"])

            pygame.draw.line(self.screen, C["border"], (x + 30, row_y + 65), (x + w - 30, row_y + 65), 1)
            row_y += row_h

    def draw_graph_panel(self, rect):
        x, y, w, h = rect
        draw_rect(self.screen, C["panel"], rect, border=C["border"])
        draw_text(self.screen, "TOTAL QUEUE OVER TIME", (x + 30, y + 20), self.fonts["h1"], C["text_main"])
        pygame.draw.line(self.screen, C["border"], (x, y + 60), (x + w, y + 60), 2)

        total_waiting = sum(len(q) for q in self.station.queues.values())
        self.queue_graph.update(self.env.now, total_waiting)

        graph_rect = (x + 50, y + 80, w - 80, h - 110)
        self.queue_graph.draw(self.screen, graph_rect, C["accent"], self.fonts["small"])

        draw_text(self.screen, f"LIVE: {total_waiting}", (x + w - 30, y + 20), self.fonts["h1"], C["accent_warn"], "topright")

    def draw_log_panel(self, rect):
        x, y, w, h = rect
        draw_rect(self.screen, C["panel"], rect, border=C["border"])
        draw_text(self.screen, "RECENT DISPATCH LOG", (x + 30, y + 20), self.fonts["h1"], C["text_main"])
        pygame.draw.line(self.screen, C["border"], (x, y + 60), (x + w, y + 60), 2)

        buses = sorted(simulation._current_run_buses, key=lambda b: b.actual_arrival, reverse=True)
        
        row_y = y + 75
        for bus in buses[:12]: 
            raw_min = int(bus.actual_arrival)
            bh = (raw_min // 60) + 6
            bm = raw_min % 60
            ap = "AM" if bh < 12 else "PM"
            bh = bh if bh <= 12 else bh - 12
            
            time_str = f"{bh:02d}:{bm:02d} {ap}"
            name_str = bus.route.replace("_", " ")[:18]
            
            is_active = (bus.actual_arrival <= self.env.now and bus.departure_time is None)
            status = "BOARDING" if is_active else ("DEPARTED" if bus.departure_time else "INCOMING")
            s_color = C["accent"] if is_active else (C["accent_ok"] if bus.departure_time else C["text_dim"])

            draw_text(self.screen, time_str, (x + 30, row_y), self.fonts["body"], C["text_dim"])
            draw_text(self.screen, name_str, (x + 150, row_y), self.fonts["body"], C["text_main"])
            draw_text(self.screen, status, (x + w - 30, row_y), self.fonts["h2"], s_color, "topright")
            
            row_y += 35

    # ─────────────────────────────────────────────────────────────────────────
    #  EXECUTION
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        while True:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                # ESC KEY EXITS THE FULLSCREEN DEMO
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

            self.screen.fill(C["bg"])

            # 1080p Grid Layout Calculations
            self.draw_header()
            self.draw_kpi_row(y_offset=110)
            
            self.draw_timeline(rect=(30, 240, WIN_W - 60, 80))
            
            main_y = 350
            main_h = WIN_H - main_y - 30
            
            left_w = 1150
            right_x = 30 + left_w + 30
            right_w = WIN_W - right_x - 30

            self.draw_routes_panel(rect=(30, main_y, left_w, main_h))
            
            graph_h = 320
            log_h = main_h - graph_h - 30
            
            self.draw_graph_panel(rect=(right_x, main_y, right_w, graph_h))
            self.draw_log_panel(rect=(right_x, main_y + graph_h + 30, right_w, log_h))
            
            # Subtle hint for the audience/presenter
            draw_text(self.screen, "[ PRESS ESC TO EXIT FULLSCREEN ]", (WIN_W // 2, WIN_H - 15), self.fonts["small"], C["border"], "midbottom")

            pygame.display.flip()

if __name__ == "__main__":
    db = PresentationDashboard()
    db.run()