import random

'''
Experiment definitions for various scenarios. Each of these returns a dictionary specifying
events that will be used in a Storyboard.
'''

def mislead_experiment(mislead, length):
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": ['0','1'], "location": ['0'], "path_type": "same"}
    event_dict[11] = {"name":"move", "actors":['1'], "location": ['1']}
    event_dict[12] = {"name": "exclusive_random", "actors": ['0','1'], "stop": 12 + mislead}
    event_dict[12 + mislead] = {"name":"move", "actors":['1'],"location":['2']}
    event_dict[12 + mislead+1] = {"name": "exclusive_random", "actors": ['0',"1"], "stop": length}
    #experiment_info = {'cross path location': loc[0], 'poi':poi, 'last':third_loc}
    return event_dict, "1"



def second_order_tom_experiment(mislead, length):
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": ['0','1','2'], "location": ['0'], "path_type":"same"}
    event_dict[16] = {"name": "cross_paths","actors": ['1','2'], "location": ['1'], "path_type":"same", "prev": ['0'], "exclude": ['0']}
    event_dict[17] = {"name": "exclusive_random", "actors": ['0','1','2'], "stop": 17 + mislead}
    event_dict[17 + mislead] = {"name":"move", "actors":['2'],"location":['2']}
    event_dict[17 + mislead + 1] = {"name": "exclusive_random", "actors": ['0','1','2'], "stop": length}
    # experiment_info = {'cross path locs': [loc_1, loc_2], 'poi': poi, 'last': loc_3}
    return event_dict, "2"

def third_order_tom_experiment(mislead, length):
    event_dict = {}
    event_dict[10] = {"name": "cross_paths","actors": ['0','1','2','3'], "location": ['0'], "path_type":"same"}
    event_dict[18] = {"name": "cross_paths","actors": ['1','2','3'], "location": ['1'], "path_type":"same", "prev": ['0'], "exclude": ['0']}
    event_dict[24] = {"name": "cross_paths","actors": ['2','3'], "location": ['2'], "path_type":"same", "prev": ['0', '1'], "exclude": ['0','1']}
    event_dict[25] = {"name": "exclusive_random", "actors": ['0','1','2', '3'], "stop": 25 + mislead}
    event_dict[25 + mislead] = {"name":"move", "actors":['3'],"location":['3']}
    event_dict[25 + mislead + 1] = {"name": "exclusive_random", "actors": ['0','1','2', '3'], "stop": length}
    # experiment_info = {'cross path locs': [loc_1, loc_2], 'poi': poi, 'last': loc_3}
    return event_dict, "3"
    
def painting_experiment(length):
    event_dict = {}
    event_dict[0] = {"name": "exclusive_random", "actors": ['0','1'], "stop": 15}
    event_dict[15]  = {"name":"move", "actors": ['0'], "location": ["puts his painting on the wall"]}
    event_dict[16]  = {"name":"move", "actors": ['1'], "location": ["enters the room"]}
    event_dict[17] = {"name": "exclusive_random", "actors": ['0','1'], "stop": 28}
    event_dict[28]  = {"name":"move", "actors": ['1'], "location": ["exclaims that the painting is horrible"]}
    event_dict[29]  = {"name":"move", "actors": ['1'], "location": ["greets everyone in the room and they all smile back at him. Then he leaves"]}
    return event_dict, "1"

def painting_experiment_2(story_length):
    manual_actions = dict()
    art_opinion = ['he hates art', 'is secretly uncomfortable around paintings', 'is anxious around 0', "thinks museums and art are boring"]
    manual_actions[0] = {'action': f'1 is trying to impress 0 with his art knowledge, but {random.choice(art_opinion)}'}
    manual_actions[story_length-1] = {'action': f"But it's late so both 0 and 1 decide to go home."}
    return {}, manual_actions, "1"
    
def painting_dislike(story_length):
    manual_actions = dict()
    event_dict = dict()
    event_dict[5] =  {"name":"cross_paths", "actors": ['0', '1'], "location": ["0"], 'path_type': "same"}
    reaction = random.choice(["scoffs at the painting and mutters under his breath", 
                            "shakes their head and walks away from the painting",
                            "rolls his eyes at the painting",
                            "wrinkles his face while looking at the painting",
                            "laughs to himself in disbelief at the painting",
                            "squints at the painting and tilts his head in disapproval",
                            "grimaces at the painting before moving on",
                            "stares blankly at the painting and shrugs",
                            "snorts quietly while looking at the painting"])
    explicit_reaction = random.choice(["shares a scathing critique of the painting with 1",
                                        "asks 1 how anyone could like this painting",
                                        "remarks to 1 that the painting is a waste of wall space",
                                        "tells 1 that looking at this piece is a waste of time"])
    
    manual_actions[6] = {"action": f"0 {explicit_reaction}"}
    event_dict[12] =  {"name":"cross_paths", "actors": ['1', '2'], "location": ["0"], 'path_type': "same"}
    manual_actions[13] = {"action": f"2 and 1 both agree that the piece in front of them is unique"}
    return event_dict, manual_actions, "2", "0"


# Deprecated
def cross_path_overlap(actors, locs, g, mislead, length, n):
    poi = random.sample(actors, n)
    loc = random.sample(locs, 1)
    second_loc = random.choice(g[loc[0]])
    event_dict = {}
    event_dict[15] = {"name": "cross_paths","actors": poi, "location": loc, "path_type": "same"}
    event_dict[16] = {"name":"move", "actor":poi[-1], "location": second_loc}
    event_dict[17] = {"name": "exclusive_random", "actors": poi, "stop": 17 + mislead}
    event_dict[17 + mislead] = {"name": "mislead", "actors": poi}
    event_dict[17 + mislead+1] = {"name": "exclusive_random", "actors": poi, "stop": length}
    experiment_info = {'cross path location': loc[0], 'poi':poi}
    return event_dict, second_loc, experiment_info

def sally_anne(n):
    event_dict = {}
    event_dict[4] = {"name": "cross_paths","actors": ['0','1'], "location": ['0'], "path_type": "same"}
    event_dict[5] = {"name":"move", "actors":['1'], "location": ['1']}
    event_dict[6] = {"name": "exclusive_random", "actors": ['0','1'], "stop": n}
    manual_actions_dict = {}
    obj = random.sample(["marble", "figurine", "doll"], 2)
    manual_actions_dict[4] = {'action': f'1 places a {obj[0]} in the basket'}
    manual_actions_dict[5] = {'action':f'0 empties the basket and places a {obj[1]} inside'}
    # experiment_info = {'cross path location': ['0'], 'poi':['0','1'], "obj": obj[0]}
    return event_dict, manual_actions_dict, "2", obj[0]
# TODO: manual_actions?

def goal_oriented(n):
    event_dict = {}
    manual_actions_dict = {}
    manual_actions_dict[0] = {'action': f'1 is trying to get away from 0'}
    event_dict[10] = {"name": "cross_paths","actors": ['0','1'], "location": ['0'], "path_type": "different"}
    event_dict[20] = {"name": "cross_paths","actors": ['0','1'], "location": ['0'], "path_type": "same"}
    return event_dict, manual_actions_dict, "1"