import numpy as np
np.random.seed(0)

class SparseGraph:
    def __init__(self):
        self.graph = {}

    def add_node(self, node_id, attributes=None):
        if node_id not in self.graph:
            self.graph[node_id] = set()

    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.graph[node1].add(node2)
        
    def get_all_node_ids(self):
        return list(self.graph.keys())

    def get_neighbors(self, node_id):
        return self.graph.get(node_id, set())


def calculate_seq_per_poi(dataset,t_day,t_granularity):
    checkin_seq_per_poi={}
    for uid,seq in enumerate(dataset):
        checkin_hour = (np.array(seq['arrival_times'])*t_granularity).astype(int)
        checkin_time = seq['arrival_times']
        for pid,poi in enumerate(seq['checkins']):
            if poi not in checkin_seq_per_poi.keys():
                checkin_seq_per_poi[poi] = {str(i+1) : [] for i in range(t_day)}
            checkin_seq_per_poi[poi][str((checkin_hour[pid]//t_granularity)+1)].append((uid,checkin_time[pid]))
    return checkin_seq_per_poi

def construct_network_for_epidemic_simulation(poi_checkin_data,t_day):
    graphs = [SparseGraph() for i in range(t_day)]
    for poi in poi_checkin_data.keys():
        for i in range(t_day):
            seq_ids_time_list = poi_checkin_data[poi][str(i+1)]
            sorted_seq_ids_time_list = sorted(seq_ids_time_list, key=lambda x: x[1])
            if len(sorted_seq_ids_time_list) >1:
                for j in range(len(sorted_seq_ids_time_list)):
                    for k in range(len(sorted_seq_ids_time_list)):
                        if sorted_seq_ids_time_list[j][0] != sorted_seq_ids_time_list[k][0]:
                            graphs[i].add_edge(sorted_seq_ids_time_list[j][0],sorted_seq_ids_time_list[k][0])
                            graphs[i].add_edge(sorted_seq_ids_time_list[k][0],sorted_seq_ids_time_list[j][0])
    return graphs

class Global_epidemic_info:
    def __init__(self,is_covid19):
        #close_contact_ratio,transmission_period,incubation_period,infection_period,reproduction_rate
        if is_covid19 :
            self.c = 0.2
            self.T = 5.8
            self.T_i = 5.2
            self.T_f = 11
            self.R_0 = 2.2
            self.beta = self.R_0 / self.T
            self.alpha = 1 / self.T_i
            self.r = 1 / self.T_f
        else:
            self.c = 0.2
            self.beta = 0.402
            self.alpha = 0.526
            self.r = 0.244

        self.susceptible = 0
        self.exposed = 50
        self.infected = 0
        self.recovered = 0

        self.susceptible_list = [self.susceptible]
        self.exposed_list = [self.exposed]
        self.infected_list = [self.infected]
        self.recovered_list = [self.recovered]

        self.susceptible_user =[]
        self.exposed_user = []
        self.infected_user = []
        self.recovered_user = []

    def update(self):
        self.susceptible_list.append(len(self.susceptible_user))
        self.exposed_list.append(len(self.exposed_user))
        self.infected_list.append(len(self.infected_user))
        self.recovered_list.append(len(self.recovered_user))

def epidemic_simulator(G,init_exposed_num,cycles,is_covid19):
    ramdom_matrix = np.random.rand(10000000)
    ramdom_index = 0
    epidemic_record = Global_epidemic_info(is_covid19=is_covid19)
    day1_user_list = list(G[0].get_all_node_ids())
    exposed_user_init = np.random.choice(day1_user_list, size=init_exposed_num, replace=False).tolist()
    epidemic_record.exposed_user.extend(exposed_user_init)
    epidemic_record.exposed_user = list(set(epidemic_record.exposed_user))
    epidemic_record.susceptible_user = []

    for cycle in range(cycles):#for one cycle, the exposed,susceptible,infected and revovered may not converged, we need multiple cycles
        for i in range(len(G)):
            susceptible_new = []
            for exposed_u in epidemic_record.exposed_user:
                susceptible_new.extend(list(G[i].get_neighbors(exposed_u)))   
            for infected_u in epidemic_record.infected_user:
                susceptible_new.extend(list(G[i].get_neighbors(infected_u)))         
            susceptible_new = list(set(susceptible_new))
            susceptible_used = []
            for u in susceptible_new:
                if (u in epidemic_record.infected_user) or (u in epidemic_record.exposed_user) or (u in epidemic_record.recovered_user):
                    pass
                else :
                    susceptible_used.append(u)
            epidemic_record.susceptible_user.extend(susceptible_used)
            epidemic_record.susceptible_user = list(set(epidemic_record.susceptible_user))
            #susceptible -> exposed
            for susceptible_u in epidemic_record.susceptible_user:
                if ramdom_matrix[ramdom_index] <= (epidemic_record.c*epidemic_record.beta):
                    epidemic_record.susceptible_user.remove(susceptible_u)
                    epidemic_record.exposed_user.append(susceptible_u)
                ramdom_index+=1
            #exposed -> infected
            for exposed_u in epidemic_record.exposed_user:
                if ramdom_matrix[ramdom_index] <= epidemic_record.alpha:
                    epidemic_record.exposed_user.remove(exposed_u)
                    epidemic_record.infected_user.append(exposed_u)
                ramdom_index+=1
            #infected -> revovered
            for infected_u in epidemic_record.infected_user:
                if ramdom_matrix[ramdom_index] <= epidemic_record.r:
                    epidemic_record.recovered_user.append(infected_u)
                    epidemic_record.infected_user.remove(infected_u)
                ramdom_index+=1
            epidemic_record.update()
            epidemic_record.susceptible_user=[]
    return epidemic_record.exposed_list, epidemic_record.infected_list, epidemic_record.recovered_list



def run_simulator(datalist,init_exposed_num,exp_num=15,cycles=12,is_covid19=True,t_day=7,t_granularity=24):
    all_data_result_exposed=[]
    all_data_result_infected=[]
    all_data_result_recovered=[]

    for idx,data in enumerate(datalist):
        checkin_seq_per_poi = calculate_seq_per_poi(data,t_day,t_granularity)
        G = construct_network_for_epidemic_simulation(checkin_seq_per_poi,t_day)
        result_exposed = []
        result_infected = []
        result_recovered = []
        for i in range(exp_num):
            exposed, infected, recovered=epidemic_simulator(G,init_exposed_num,cycles,is_covid19)
            result_exposed.append(exposed[1:])
            result_infected.append(infected[1:])
            result_recovered.append(recovered[1:])
        result_exposed = np.mean(np.array(result_exposed), axis=0)
        result_infected = np.mean(np.array(result_infected), axis=0)
        result_recovered = np.mean(np.array(result_recovered), axis=0)

        all_data_result_exposed.append(result_exposed)
        all_data_result_infected.append(result_infected)
        all_data_result_recovered.append(result_recovered)

    real_exposed = all_data_result_exposed[0]
    generated_exposed = all_data_result_exposed[1]

    real_infected = all_data_result_infected[0]
    generated_infected = all_data_result_infected[1]

    real_recovered = all_data_result_recovered[0]
    generated_recovered = all_data_result_recovered[1]

    relative_difference_e = np.abs((real_exposed - generated_exposed) / (real_exposed))
    relative_difference_i = np.abs((real_infected - generated_infected) / (real_infected))
    relative_difference_r = np.abs((real_recovered - generated_recovered) / (real_recovered))

    return relative_difference_e,relative_difference_i,relative_difference_r

def run_EpiSim_task(test_data, generated_data, init_exposed_num=50, exp_num = 15, cycles = 12):
    datalist = [test_data,generated_data]
    relative_difference_e_Covid19,relative_difference_i_Covid19,relative_difference_r_Covid19 = run_simulator(datalist,init_exposed_num,exp_num,cycles,True)
    relative_difference_e_Influenza,relative_difference_i_Influenza,relative_difference_r_Influenza = run_simulator(datalist,init_exposed_num,exp_num,cycles,False)
    results_relative = np.array([relative_difference_e_Covid19,relative_difference_i_Covid19,relative_difference_r_Covid19,\
                                 relative_difference_e_Influenza,relative_difference_i_Influenza,relative_difference_r_Influenza])
    MAPE = np.mean(results_relative)
    MSPE = np.mean(np.square(results_relative))

    return MAPE,MSPE