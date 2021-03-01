import logging


class Weight_CBN(object):
    def __init__(self, entities, bridge_rank, KGE_rank, true_rank):
        self.entities = entities
        self.bridge_rank = bridge_rank
        self.kge_rank = KGE_rank
        self.true_rank = true_rank

    def takeSecond(elem):
        return elem[1]

    def weighted(self, main_rank, add_rank):
        main_weight_len = len(main_rank)
        add_weight_len = len(add_rank)
        main_weight = 0.5
        add_weight = 0.5
        new_rank = list()

        for entity in main_rank:
            entity_weight1 = (main_weight_len-main_rank.index(entity))/main_weight_len*main_weight
            entity_weight2 = (add_weight_len-add_rank.index(entity))/add_weight_len*add_weight
            sum_entity = entity_weight1 + entity_weight2
            new_couple = (entity, sum_entity)
            new_rank.append(new_couple)

        new_rank.sort(key=self.takeSecond, reverse=True)

        return new_rank
    
    def calculate_rank(self, all_main_rank, all_add_rank, true_rank):
        rank_list = list()
        for true in true_rank:
            number = true_rank.index(true)
            main_rank = all_main_rank[number]
            add_rank = all_add_rank[number]
            newrank = Weight_CBN.weighted(main_rank, newrank)
            rank_list.append(Weight_CBN.ranking(newrank, true))
        return rank_list
            
    
    def ranking(self, score_list, true_entity):
        
        number = 1
        for entity in score_list:
            if entity == true_entity:
                break
            else:
                number += 1
        return number
    
    def metric(self, rank_list):
        MRR_all = 0
        MR_all = 0
        hit_1 = 0
        hit_3 = 0
        hit_10 = 10
        for rank in rank_list:
            if rank <=1:
                hit_1 += 1
                MR_all += rank
                MRR_all += 1/rank
            elif rank <= 3:
                hit_3 += 1
                MR_all += rank
                MRR_all += 1 / rank
            elif rank <= 10:
                hit_10 += 1
                MR_all += rank
                MRR_all += 1 / rank
            else:
                MR_all += rank
                MRR_all += 1 / rank
        num_entities = len(rank_list)
        MRR = MRR_all/num_entities
        MR = MR_all/num_entities
        hit_1 = hit_1/num_entities
        hit_3 = hit_3/num_entities
        hit_10 = hit_10/num_entities
        
        return MRR, MR, hit_1, hit_3, hit_10

    def main_log(self):
        rank_list = self.calculate_rank(self.kge_rank, self.bridge_rank, self.true_rank)
        MRR, MR, Hit_1, Hit_3, Hit_10 = self.metric(rank_list)
        logging.info("MRR:%f, MR:%d, Hit_1:%f, Hit_3:%f, Hit_10:%f" % (MRR, MR, Hit_1, Hit_3, Hit_10))


