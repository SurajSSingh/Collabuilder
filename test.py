import MalmoPython
import time
import json

MAX_RETRIES = 5
MISSION_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Do stuff!</Summary>
              </About>

              <ModSettings>
                <MsPerTick>50</MsPerTick>
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
                  <FlatWorldGenerator generatorString="3;3*minecraft:dirt;3*minecraft:bedrock;1;"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>Blockhead</Name>
                <AgentStart>
                  <Placement x="0.5" y="4.0" z="0.5" yaw="0" pitch="70"/>
                  <Inventory>
                    <InventoryBlock slot="0" type="stone" quantity="64"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <DiscreteMovementCommands/>
                  <RewardForSendingCommand reward="-1" />
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

def make_ah():
    agent_host = MalmoPython.AgentHost()

    my_mission = MalmoPython.MissionSpec(MISSION_XML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start a mission:
    for retry in range(MAX_RETRIES):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == MAX_RETRIES - 1:
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
        for error in world_state.errors:
            print("Error:",error.text)

    print("\nMission running.")
    return agent_host

if __name__ == '__main__':
    ah = make_ah()
    import pdb; pdb.set_trace()
    input()