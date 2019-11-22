import sys
import MalmoPython
import json

import time
from collections import namedtuple

from world_model import WorldModel

Mission = namedtuple('Mission', [
        'blueprint',
        'start_position',
        'training',
        'action_delay',
        'max_episode_time',
        'simulated',
        'display'
    ])

MissionStats = namedtuple('MissionStats', [
        'reward',
        'length'
    ])

_AGENT_HOST = None

def run_mission(model, mission, cfg, demo=False):
    global _AGENT_HOST

    if mission.simulated:
        return run_simulated_mission(model, mission, cfg, demo=demo)

    # Only import MalmoPython and set up the agent_host if we're actually using them
    import MalmoPython
    if _AGENT_HOST is None:
        _AGENT_HOST = MalmoPython.AgentHost()
        try:
            _AGENT_HOST.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:',e)
            print(_AGENT_HOST.getUsage())
            exit(1)
        if _AGENT_HOST.receivedArgument("help"):
            print(_AGENT_HOST.getUsage())
            exit(0)

    return run_malmo_mission(model, mission, _construct_xml(mission,cfg), cfg, _AGENT_HOST, demo=demo)

def run_malmo_mission(model, mission, mission_xml, cfg, agent_host, max_retries=5, demo=False):
    # Create default Malmo objects:
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    world_model = WorldModel(mission.blueprint, cfg, simulated=False)
    # Attempt to start a mission:
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2**retry)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        time.sleep(1)
        for error in world_state.errors:
            print("Error:",error.text)

    print("\nMission running.")

    total_reward = 0
    current_r = 0

    start = time.time()
    # Loop until mission ends
    while (world_state.is_mission_running and
           world_model.is_mission_running()):
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
        current_r += sum(r.getValue() for r in world_state.rewards)
        if len(world_state.observations) > 0:
            raw_obs = json.loads(world_state.observations[-1].text)
            world_model.update(raw_obs)
            current_r += world_model.reward()
            if demo:
                action = model.demo_act( world_model.get_observation() )
            else:
                action = model.act( current_r, world_model.get_observation() )
            if mission.display is not None:
                mission.display.update(world_model)
            total_reward += current_r
            current_r = 0
            if world_model.mission_complete() or not world_model.agent_in_arena():
                agent_host.sendCommand('quit')
            elif world_state.is_mission_running:
                agent_host.sendCommand( action )
                if demo:
                    print(action)
        time.sleep(mission.action_delay)
    end = time.time()

    model.mission_ended()

    print()
    print("Mission ended")

    return MissionStats(
            reward = total_reward,
            length = end - start
        )

def run_simulated_mission(model, mission, cfg, demo=False):
    print("Simulated mission running.")

    world_model  = WorldModel(mission.blueprint, cfg, simulated=True, agent_pos=mission.start_position)
    ticks_left   = 5*mission.max_episode_time
    total_reward = 0
    current_r    = 0
    use_delays   = mission.action_delay > 0

    while (ticks_left > 0 and
           world_model.is_mission_running()):
        ticks_left -= 1
        current_r = world_model.reward()
        if demo:
            action = model.demo_act(world_model.get_observation())
        else:
            action = model.act(current_r, world_model.get_observation())
        if mission.display is not None:
            mission.display.update(world_model)
        total_reward += current_r
        world_model.simulate(action)
        if use_delays:
            print(action)
            time.sleep(mission.action_delay)

    # Collect last reward, and give to model, then end the mission
    if mission.display is not None:
        mission.display.update(world_model)
    current_r = world_model.reward()
    if not demo:
        model.act(current_r, world_model.get_observation())
    total_reward += current_r
    model.mission_ended()
    print("Simulated mission ended")

    return MissionStats(
            reward = total_reward,
            length = (mission.max_episode_time - (ticks_left/5))
        )

def _construct_xml(mission, cfg):
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

            <About>
              <Summary>Place a block!</Summary>
            </About>

              <ModSettings>
                <MsPerTick>{ms_per_tick}</MsPerTick>
                <PrioritiseOffscreenRendering>{offscreen_rendering}</PrioritiseOffscreenRendering>
              </ModSettings>

            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>12000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;5*minecraft:bedrock;1;" forceReset="1"/>
                  <ServerQuitFromTimeUp timeLimitMs="{server_timeout}"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Blockhead</Name>
                <AgentStart>
                  <Placement x="{start_x}" y="{start_y}" z="{start_z}" yaw="0" pitch="70"/>
                  <Inventory>
                    <InventoryObject slot="0" type="stone" quantity="64"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ObservationFromGrid>
                    <Grid name="world_grid" absoluteCoords="1">
                      <min x="{arena_x1}" y="{arena_y1}" z="{arena_z1}"/>
                      <max x="{arena_x2}" y="{arena_y2}" z="{arena_z2}"/>
                    </Grid>
                  </ObservationFromGrid>
                  <DiscreteMovementCommands/>
                  <MissionQuitCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''.format(
                    ms_per_tick         = int(50/cfg('training', 'overclock_factor') if mission.training else 50),
                    offscreen_rendering = ('false' if mission.training else 'true'),
                    start_x = mission.start_position[0] + cfg('arena', 'anchor', 'x') + cfg('arena', 'offset', 'x'),
                    start_y = mission.start_position[1] + cfg('arena', 'anchor', 'y') + cfg('arena', 'offset', 'y') + 1, # +1 corrects for mismatch in start positioning vs. reading position back
                    start_z = mission.start_position[2] + cfg('arena', 'anchor', 'z') + cfg('arena', 'offset', 'z'),
                    arena_x1 = cfg('arena', 'anchor', 'x'),     arena_x2 = cfg('arena', 'anchor', 'x') - 1 + cfg('arena', 'width'),
                    # -1 here reads the floor as well, to detect attacking the floor
                    arena_y1 = cfg('arena', 'anchor', 'y') - 1, arena_y2 = cfg('arena', 'anchor', 'y') - 1 + cfg('arena', 'height'),
                    arena_z1 = cfg('arena', 'anchor', 'z'),     arena_z2 = cfg('arena', 'anchor', 'z') - 1 + cfg('arena', 'length'),
                    server_timeout = 1000*mission.max_episode_time
                )
