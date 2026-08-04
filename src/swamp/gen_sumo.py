"""P3: Generate SUMO arterial network files (Peachtree-like, 4 intersections)."""
import os, subprocess

OUT = "/data/lab/swamp/runs/p3"
os.makedirs(OUT, exist_ok=True)
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
NETCONVERT = os.path.join(SUMO_HOME, "bin", "netconvert")

INT_POS = [0, 160, 320, 480, 640]
CROSS = 80
CYCLE = 100
G_THRU, Y_DUR = 40, 4
R_THRU = CYCLE - G_THRU - Y_DUR


def gen_nodes():
    p = os.path.join(OUT, "arterial.nod.xml")
    with open(p, "w") as f:
        f.write('<?xml version="1.0"?>\n<nodes>\n')
        for i, x in enumerate(INT_POS):
            f.write(f'  <node id="N{i}" x="{x}" y="0" type="traffic_light"/>\n')
            f.write(f'  <node id="CN{i}" x="{x}" y="{CROSS/2}" type="priority"/>\n')
            f.write(f'  <node id="CS{i}" x="{x}" y="{-CROSS/2}" type="priority"/>\n')
        f.write('</nodes>\n')
    return p


def gen_edges():
    p = os.path.join(OUT, "arterial.edg.xml")
    with open(p, "w") as f:
        f.write('<?xml version="1.0"?>\n<edges>\n')
        for i in range(len(INT_POS) - 1):
            f.write(f'  <edge id="E{i}" from="N{i}" to="N{i+1}" numLanes="2" speed="15"/>\n')
            f.write(f'  <edge id="W{i}" from="N{i+1}" to="N{i}" numLanes="2" speed="15"/>\n')
        for i in range(len(INT_POS)):
            f.write(f'  <edge id="CN{i}" from="N{i}" to="CN{i}" numLanes="1" speed="10"/>\n')
            f.write(f'  <edge id="SC{i}" from="CN{i}" to="N{i}" numLanes="1" speed="10"/>\n')
            f.write(f'  <edge id="CS{i}" from="N{i}" to="CS{i}" numLanes="1" speed="10"/>\n')
            f.write(f'  <edge id="SC{i}b" from="CS{i}" to="N{i}" numLanes="1" speed="10"/>\n')
        f.write('</edges>\n')
    return p


def gen_signals():
    """Coordinated signal: NB direction gets green wave, offsets staggered."""
    p = os.path.join(OUT, "arterial.add.xml")
    # phase string: 2 chars per edge connection at the node
    # For simplicity: major (arterial) green, then cross green
    # Arterial NB+SB: "GG" green, "yy" yellow, "rr" red
    # Cross: "rr" when arterial green, "GG" when arterial red
    g = "G" * 2  # 2 arterial lanes
    y = "y" * 2
    r = "r" * 2
    cg = "G" * 2  # cross street
    cy = "y" * 2
    cr = "r" * 2
    # Phase 1: arterial green (40s), Phase 2: arterial yellow (4s),
    # Phase 3: cross green (52s), Phase 4: cross yellow (4s)
    phases = [
        (G_THRU, g + cr),  # arterial green, cross red
        (Y_DUR, y + cr),   # arterial yellow, cross red
        (R_THRU - Y_DUR, r + cg),  # arterial red, cross green
        (Y_DUR, r + cy),   # arterial red, cross yellow
    ]
    with open(p, "w") as f:
        f.write('<?xml version="1.0"?>\n<additional>\n')
        for i in range(len(INT_POS)):
            # offset for coordination: green wave at ~15 m/s
            offset = int(i * 160 / 15) % CYCLE
            f.write(f'  <tlLogic id="N{i}" type="static" programID="coord" '
                    f'offset="{offset}">\n')
            for dur, state in phases:
                f.write(f'    <phase duration="{dur}" state="{state}"/>\n')
            f.write(f'  </tlLogic>\n')
        f.write('</additional>\n')
    return p


def gen_routes():
    """Traffic demand: NB arterial + some cross-street traffic."""
    p = os.path.join(OUT, "arterial.rou.xml")
    with open(p, "w") as f:
        f.write('<?xml version="1.0"?>\n<routes>\n')
        f.write('  <vType id="car" accel="1.5" decel="3.0" sigma="0.3" '
                'length="5" maxSpeed="15" speedFactor="0.9"/>\n')
        f.write('  <vType id="cav" accel="3.0" decel="6.0" sigma="0.0" '
                'length="5" maxSpeed="15" color="1,0,0"/>\n')
        f.write('  <route id="cav_route" edges="E0 E1 E2 E3"/>\n')
        # NB arterial flow (higher density to force signal stops)
        f.write('  <flow id="nb_flow" type="car" begin="0" end="900" '
                f'probability="0.2" from="E0" to="E3"/>\n')
        # SB arterial flow
        f.write('  <flow id="sb_flow" type="car" begin="0" end="900" '
                f'probability="0.15" from="W3" to="W0"/>\n')
        # Cross traffic at each intersection
        for i in range(1, len(INT_POS) - 1):
            f.write(f'  <flow id="cross_n{i}" type="car" begin="0" end="900" '
                    f'probability="0.05" from="SC{i}" to="CS{i}"/>\n')
        f.write('</routes>\n')
    return p


def gen_cfg():
    p = os.path.join(OUT, "arterial.sumocfg")
    with open(p, "w") as f:
        f.write('<?xml version="1.0"?>\n<configuration>\n')
        f.write('  <input>\n')
        f.write(f'    <net-file value="arterial.net.xml"/>\n')
        f.write(f'    <route-files value="arterial.rou.xml"/>\n')
        f.write(f'    <additional-files value="arterial.add.xml"/>\n')
        f.write('  </input>\n')
        f.write('  <time>\n')
        f.write(f'    <step-length value="0.1"/>\n')
        f.write(f'    <end value="1000"/>\n')
        f.write('  </time>\n')
        f.write('  <processing>\n')
        f.write('    <collision.action value="warn"/>\n')
        f.write('  </processing>\n')
        f.write('</configuration>\n')
    return p


def main():
    nod = gen_nodes()
    edg = gen_edges()
    rou = gen_routes()
    cfg = gen_cfg()
    # convert to net.xml — let netconvert auto-generate TLS programs
    net_xml = os.path.join(OUT, "arterial.net.xml")
    cmd = [NETCONVERT, "-n", nod, "-e", edg, "-o", net_xml,
           "--no-turnarounds", "true",
           "--tls.guess", "true",
           "--tls.default-type", "static"]
    print("Running netconvert:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:])
    else:
        print("Network generated:", net_xml)
    # remove add.xml from config (auto-generated TLS in net.xml)
    cfg_p = os.path.join(OUT, "arterial.sumocfg")
    with open(cfg_p, "w") as f:
        f.write('<?xml version="1.0"?>\n<configuration>\n')
        f.write('  <input>\n')
        f.write(f'    <net-file value="arterial.net.xml"/>\n')
        f.write(f'    <route-files value="arterial.rou.xml"/>\n')
        f.write('  </input>\n')
        f.write('  <time>\n')
        f.write(f'    <step-length value="0.1"/>\n')
        f.write(f'    <end value="1000"/>\n')
        f.write('  </time>\n')
        f.write('</configuration>\n')
    print("Files in", OUT, ":", os.listdir(OUT))


if __name__ == "__main__":
    main()
