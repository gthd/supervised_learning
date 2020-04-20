# from supervised import data_collector as dc
# from supervised import regression as rg
# from supervised import q_network as cl
import supervised.data_collector as dc
import supervised.regression as rg
import supervised.q_network as cl


dc.Policy().collect_training_data()
