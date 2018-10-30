from __future__ import print_function, unicode_literals

from .toolbox import distance, sigmoid, compute_median_and_mad, cosine_similarity, softmax
from .ccnews import CommonCrawlExtractor
from itertools import chain
from collections import Counter, defaultdict
from bs4 import UnicodeDammit
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import pandas as pd
import json
import ast
import pickle

import spacy
import fastText as ft
import json

import logging

import os


tqdm.monitor_interval = 0


__author__ = "Libre AI"
__copyright__ = "Copyright 2018"
__credits__ = ["SpaCy", "newsplease", "newspaper"]


class Minerva:
    __log_level = logging.INFO
    __nlp_model_name = 'en_core_web_lg'
    __nlp = spacy.load(__nlp_model_name)

    __ft_lang_model = ft.load_model("../models/lid.176.ftz")

    __risk_data = pd.read_csv("../data/entities_minerva_with_vectors.tsv", sep="\t")
    __risk2info = None
    __categories_info = None
    __categories = None

    __LANG_EN = 'en'

    __valid_hosts = ['.ie',
                     'afp.com',
                     'aljazeera.com',
                     'ap.org',
                     'asahi.com',
                     'bbc.com',
                     'chinadaily.com.cn',
                     'cnn.com',
                     'dawn.com',
                     'dw.com',
                     'efe.com',
                     'indiatimes.com',
                     'irishexaminer.com',
                     'irishtimes.com',
                     'newstalk.com',
                     'nytimes.com',
                     'reuters.com',
                     'smh.com.au',
                     'telesurtv.net',
                     'theguardian.com',
                     'usatoday.com',
                     'washingtonpost.com',
                     'wsj.com']

    def __init__(self):
        entities_minerva = json.load(open("../data/entities_minerva.json", "r"))
        self.__risk2info = {e['id']: e for e in entities_minerva}
        self.__categories = list(self.__risk_data.category.drop_duplicates())
        self.__categories_info = self.compute_category_centroid(self.__risk_data)
        self.__setup()

    def __setup(self):
        """
        Setup
        :return:
        """

        # make loggers quite
        logging.getLogger('newsplease').setLevel(logging.CRITICAL)
        logging.getLogger('spacy').setLevel(logging.CRITICAL)
        logging.getLogger('bs4').setLevel(logging.CRITICAL)
        logging.getLogger('chardet').setLevel(logging.CRITICAL)

        # set own logger
        logging.basicConfig(level=self.__log_level)
        self.__logger = logging.getLogger(__name__)

    def parallel_extract_from_ccnews(self, file_list, valid_hosts=None, n_workers=10):
        if valid_hosts:
            self.__valid_hosts = valid_hosts

        all_risk_infos = []

        files_count = 0

        for f in file_list:
            files_count += 1
            logging.info("processing file {} {}/{}".format(f, files_count, len(file_list)))

            articles = self.extract_articles_from_ccnews(f)
            logging.info("{} articles extracted".format(len(articles)))

            risk_infos = None
            with Pool(n_workers) as p:
                risk_infos = list(tqdm(p.imap(self.transform, articles), total=len(articles)))

            all_risk_infos.extend([x for x in risk_infos if x is not None])

        return list(chain.from_iterable(all_risk_infos))

    def extract_from_ccnews(self, file_list, valid_hosts=None, fname_out=None):
        if valid_hosts:
            self.__valid_hosts = valid_hosts

        all_risk_infos = []

        files_count = 0

        for f in file_list:
            files_count += 1
            logging.info("processing file {} {}/{}".format(f, files_count, len(file_list)))

            articles = self.extract_articles_from_ccnews(f)
            logging.info("{} articles extracted".format(len(articles)))

            risk_infos = list(map(self.transform, articles))

            logging.info("{} risk_infos extracted".format(len(risk_infos)))
            logging.debug(risk_infos)

            all_risk_infos.extend([x for x in risk_infos if x is not None])

        if fname_out is not None:
            with open(fname_out, "wb") as fout:
                pickle.dump(all_risk_infos, fout)

        return all_risk_infos

    def extract_and_save_articles_from_ccnews(self, file_list, valid_hosts=None, out_dir="./"):
        if valid_hosts:
            self.__valid_hosts = valid_hosts

        files_count = 0

        for f in file_list:
            files_count += 1
            logging.info("processing file {} {}/{}".format(f, files_count, len(file_list)))

            articles = self.extract_articles_from_ccnews(f)

            # dump the extracted articles:
            _, fname_out = os.path.split(f)
            fname_out = f.replace(".warc.gz", ".articles_extracted.pkl")
            Minerva.dump(articles, out_dir + os.path.sep + fname_out)
            logging.info("{} articles extracted".format(len(articles)))
            logging.info("{} saved.".format(fname_out))

    def extract_articles_from_ccnews(self, cc_gzip_file):
        ccn = CommonCrawlExtractor()
        articles = ccn.extract_from_commoncrawl_gzip_local(cc_gzip_file,
                                                           valid_hosts=self.__valid_hosts
                                                           )
        return articles

    def yield_transformed(self, article, risk_infos):
        try:
            risk_info = self.transform(article)
            if risk_info:
                risk_info_json = json.dumps(risk_info, sort_keys=True, default=str)
                risk_infos.append(risk_info_json)
        except Exception as e:
            self.__logger.exception(e)
            pass

    def save_transformed(self, article, fout):
        try:
            risk_info = self.transform(article)
            if risk_info:
                risk_info_json = json.dumps(risk_info, sort_keys=True, default=str)
                fout.write("{}\n".format(risk_info_json))
        except Exception as e:
            self.__logger.exception(e)
            pass

    def transform(self, article):
        risk_info = None
        try:
            article_info = article.get_dict()

            lang = None
            proba = 1.0

            if article_info['title'] is not None:
                article_info['title'] = self.clean_text(article_info['title'])
                lang, proba = self.detect_language(article_info['title'])
                # analyzing english articles only for the moment
                # (need to experiment more with multi-lang support of nlp libs)
                if lang != self.__LANG_EN:
                    return None
            else:
                return None

            # analyzing english articles only for the moment
            # (need to experiment more with multi-lang support of nlp libs)
            article_info['text'] = self.clean_text(article_info['text'])

            doc_vec = self.get_vector(article_info['title'])

            is_risk = self.is_risk(doc_vec)

            if is_risk:
                ents = self.extract_entities_set(article_info)

                max_risk, max_risk_sim, max_risk_category = self.compute_max_risk(doc_vec)

                if max_risk is None or max_risk_sim < 0:
                    return None

                # example: "date_publish": "2018-07-25 09:20:13"
                year = article_info['date_publish'].strftime("%Y")
                week = article_info['date_publish'].strftime("%W")

                risk_info = {'title': article_info['title'],
                             'source_domain': article_info['source_domain'],
                             'url': article_info['url'],
                             'image_url': article_info['image_url'],
                             'risk': max_risk,
                             'risk_sim': max_risk_sim,
                             'category': max_risk_category,
                             'date_publish': article_info['date_publish'],
                             'year': year,
                             'week': week,
                             'entities': ents
                             }
            else:
                risk_info = None

        except Exception as e:
            self.__logger.exception(e)
            pass

        return risk_info

    def compute_max_risk(self, doc_vec):
        max_risk = None
        max_risk_sim = -1
        max_risk_category = None

        for risk in self.__risk_data.itertuples(index=True, name="Risk"):
            risk_vec = np.array(ast.literal_eval(risk.vector))
            sim = cosine_similarity(risk_vec, doc_vec)
            if sim > max_risk_sim:
                max_risk_sim = sim
                max_risk = risk.id
                max_risk_category = risk.category
        return max_risk, max_risk_sim, max_risk_category

    def extract_article_info(self, article):
        article_info = article.get_dict()
        article_info['title'] = self.clean_text(article_info['title'])
        article_info['text'] = self.clean_text(article_info['text'])

        doc_vec = self.get_vector(article_info['title'])

        is_risk, category_risk_probas = self.is_risk(doc_vec)

        if is_risk:
            ents = self.extract_entities(article_info)
            article_info['ents'] = ents

            keys = sorted(category_risk_probas.keys())

            probas_array = np.array([category_risk_probas[k] for k in keys])
            probas_softmax = softmax(probas_array)
            index_max = np.argmax(probas_softmax)
            category_with_max_prob = keys[index_max]

            candidate_risks_df = set(self.__risk_data[self.__risk_data.category == category_with_max_prob].id)

            category_risk_probas_softmaxed = dict(zip(keys, probas_softmax))
            article_info['category_risk_probas'] = category_risk_probas_softmaxed

            risk_info = []

            max_risk = None
            max_risk_sim = -1

            for risk in self.__risk_data.itertuples(index=True, name="Risk"):
                risk_vec = np.array(ast.literal_eval(risk.vector))
                sim = cosine_similarity(risk_vec, doc_vec)

                if risk.id in candidate_risks_df:
                    if sim > max_risk_sim:
                        max_risk_sim = sim
                        max_risk = risk.id

                risk_info.append({'risk_id':risk.id, 'risk_sim':sim, 'risk_category': risk.category,
                                  'cagory_risk_prob': category_risk_probas_softmaxed[risk.category]})

            article_info['risk_info'] = risk_info

            article_info['risk'] = max_risk
            article_info['risk_category'] = category_with_max_prob

            print(json.dumps(article_info, sort_keys=True, default=str, indent=4))
        else:
            article_info = None
        return article_info

    def extract_entities(self, article_info, valid_entity_labels={'PERSON', 'NORP', 'ORG', 'GPE',
                                                                  'LOC', 'PRODUCT', 'EVENT'}):
        text = u'{} {}'.format(article_info['title'], article_info['text'])
        doc = self.__nlp(text)
        ents = doc.ents
        filtered_ents = filter(lambda w: w.label_ in valid_entity_labels, ents)
        ent_name_and_label = Counter([(e.text, e.label_) for e in filtered_ents])
        ent_name_label_count = [{'ent': ent, 'label': label, 'freq': freq} for ((ent, label), freq) in
                                ent_name_and_label.items()]
        return ent_name_label_count

    def extract_entities_set(self, article_info, valid_entity_labels={'PERSON', 'NORP', 'ORG', 'GPE',
                                                                      'LOC', 'PRODUCT', 'EVENT'}):
        text = u'{} {}'.format(article_info['title'], article_info['text'])
        doc = self.__nlp(text)
        ents = doc.ents
        filtered_ents = filter(lambda w: w.label_ in valid_entity_labels, ents)
        entitites_set = set([(e.text, e.label_) for e in filtered_ents])
        return entitites_set

    def clean_text(self, s):
        s = UnicodeDammit(s).unicode_markup
        punctuation = '\!"#$%&()*+,-./:;<=>?@[]^_`{|}~' + "\n\t\r"
        translator = str.maketrans(punctuation, ' ' * len(punctuation))
        s = s.translate(translator)
        return self.remove_redundant_spaces(s)

    def remove_redundant_spaces(self, s):
        tokens = [t.strip() for t in s.split()]
        return " ".join(tokens)

    def risk_proba(self, vec, category_centroid, median, mad, max_deviations=3, b=1.4826):
        """
        Leys, Christophe, Christophe Ley, Olivier Klein, Philippe Bernard, and Laurent Licata. 2013.
        "Detecting Outliers: Do Not Use Standard Deviation Around the Mean, Use Absolute Deviation Around the Median."
        Journal of Experimental Social Psychology 49 (4): 764–766.
        """
        # b = 1.4826 #is a constant linked to the assumption of normality of the data
        d = distance(category_centroid, vec)
        MAD = b * mad
        absolute_deviation_from_median = np.sqrt((d - median) ** 2)
        mad_deviations = absolute_deviation_from_median / MAD
        # is the deviation within the max deviations (not outlier):
        return sigmoid(max_deviations - mad_deviations)

    def get_category_vectors(self, category, risks_df):
        return np.array([np.array(ast.literal_eval(x)) for x in list(risks_df[risks_df.category == category].vector)])

    def compute_category_centroid(self, risks_df):
        categories = list(risks_df.category.drop_duplicates())
        categories_info = dict()
        for category in categories:
            category_vectors = self.get_category_vectors(category, risks_df)
            centroid = np.mean(category_vectors, axis=0)
            median, mad = compute_median_and_mad(centroid, category_vectors)
            categories_info[category] = {'centroid': centroid, 'median': median, 'mad': mad}
        return categories_info

    def is_risk(self, doc_vec, threshold=0.5, max_deviations=5):
        """check if the text is considered a risk in at least one category, if so, we process the article"""
        is_risk = False
        category_risk_probas = dict()
        for category in self.__categories:
            categories_info = self.__categories_info[category]
            risk_proba = self.risk_proba(doc_vec,
                                         categories_info['centroid'],
                                         categories_info['median'],
                                         categories_info['mad'],
                                         max_deviations=max_deviations)
            category_risk_probas[category] = risk_proba
            is_risk = is_risk or (risk_proba > threshold)
        return is_risk

    def get_vector(self, w):
        x = self.__nlp(w)
        norm = x.vector_norm
        norm = norm if norm > 0 else 1.0
        normalized_vec = x.vector / norm
        return normalized_vec

    # TODO: Check!. Computing the vec in this way is slow, but could lead to a better representation
    # due to the micro-avergaing at the level of the entity composed by several words

    # def get_vector_from_list(self, entities):
    #     vectors = [self.get_vector(e) for e in entities]
    #
    #     # representing the global risk as the mean of the individual vectors
    #     vec = np.mean(vectors, axis=0)
    #
    #     # normalizing to unit length
    #     vec_norm = np.linalg.norm(vec)
    #     vec_norm = vec_norm if vec_norm > 0 else 1.0
    #     vec_normalized = vec / vec_norm
    #
    #     return vec_normalized

    def get_vector_from_list(self, entities):
        # vector of bag of entities
        vec = self.__nlp(" ".join(entities)).vector

        # normalizing to unit length
        vec_norm = np.linalg.norm(vec)
        vec_norm = vec_norm if vec_norm > 0 else 1.0
        vec_normalized = vec / vec_norm

        return vec_normalized

    def compute_global_risk_vectors(self, risk):
        vectors = []

        # type and risk category:
        type_category = "{} {}".format(risk.type, risk.category)
        vec_type_category = self.get_vector(type_category)
        vectors.append(vec_type_category)

        # long label:
        vec_long_label = self.get_vector(risk.long_label)
        vectors.append(vec_long_label)

        # description:
        vec_desc = self.get_vector(risk.description)
        vectors.append(vec_desc)

        # the keywords:
        for keyword in [k.strip() for k in risk.keywords.strip().split(",")]:
            keyword_vec = self.get_vector(keyword)
            vectors.append(keyword_vec)

        return vectors

    def compute_global_risk_vector(self, risk):
        vectors = self.compute_global_risk_vectors(risk)

        # representing the global risk as the mean of the individual vectors
        vec_global_risk = np.mean(vectors, axis=0)

        # normalizing to unit length
        vec_norm = np.linalg.norm(vec_global_risk)
        vec_norm = vec_norm if vec_norm > 0 else 1.0
        vec_normalized = vec_global_risk / vec_norm

        return vec_normalized

    def detect_language(self, text):
        label, proba = self.__ft_lang_model.predict(text)
        label = label[0].replace('__label__', '')
        proba = proba[0]
        return label, proba

    def filter_articles(self, articles):
        return [a for a in articles if a.get_dict()['text'] is not None]

    def filter_pickled_articles(self, file_list):
        all_articles = []
        for f in file_list:
            with open(f, "rb") as fin:
                articles = pickle.load(fin)
                all_articles.extend(self.filter_articles(articles))
        return all_articles

    def extract_risk_entities_over_time(self, risk_infos):
        time2risk = dict()
        for info in risk_infos:
            year_week = (int(info['year']), int(info['week']))

            if year_week not in time2risk:
                time2risk[year_week] = dict()

            if info['risk'] not in time2risk[year_week]:
                time2risk[year_week][info['risk']] = Counter()

            time2risk[year_week][info['risk']].update(info['entities'])
        return time2risk

    def extract_risk_entities(self, risk_infos):
        risk2entities = dict()
        for info in risk_infos:
            if info['risk'] not in risk2entities:
                risk2entities[info['risk']] = Counter()

            risk2entities[info['risk']].update(info['entities'])
        return risk2entities

    def compute_link_weights_jaccard(self, risks):
        risk_keys = sorted(list(risks.keys()))
        links = []
        for i in range(0, len(risk_keys)):
            for j in range(i + 1, len(risk_keys)):
                ri = risk_keys[i]
                rj = risk_keys[j]
                entities_i = [x[0] for x in risks[ri].keys()]
                entities_j = [x[0] for x in risks[rj].keys()]
                wij = self.jaccard_sim(set(entities_i), set(entities_j))
                links.append((ri, rj, wij))
        return links

    def compute_link_weights(self, risks):
        risk_keys = sorted(list(risks.keys()))
        links = []
        for i in range(0, len(risk_keys)):
            for j in range(i + 1, len(risk_keys)):
                ri = risk_keys[i]
                rj = risk_keys[j]
                # wij = self.jaccard_sim(set(risks[ri].keys()), set(risks[rj].keys()))
                # first element, i.e., index 0, of the tuple is the entity name and the second element the type,
                # we do not use the entity type to compute the latent vector
                entities_i = [x[0] for x in risks[ri].keys()]
                entities_j = [x[0] for x in risks[rj].keys()]
                wij = self.latent_entities_similarity(entities_i, entities_j)
                links.append((ri, rj, wij))
        return links

    def compute_link_weights_over_time(self, time2risk):
        time2risk_links = dict()
        for t in time2risk.keys():
            time2risk_links[t] = self.compute_link_weights(time2risk[t])
        return time2risk_links

    def compute_topn_links(self, risk_links, topn=8):
        risk2links = dict()
        for ri, rj, wij in risk_links:
            if ri not in risk2links:
                risk2links[ri] = Counter()

            if rj not in risk2links:
                risk2links[rj] = Counter()

            risk2links[ri][rj] += wij
            risk2links[rj][ri] += wij

        topn_links = dict()
        for risk in risk2links.keys():
            topn_links[risk] = risk2links[risk].most_common(topn)
        return topn_links

    def jaccard_sim(self, a, b):
        return len(a.intersection(b)) / len(a.union(b))

    def latent_entities_similarity(self, a, b):
        vec_a = self.get_vector_from_list(a)
        vec_b = self.get_vector_from_list(b)
        return cosine_similarity(vec_a, vec_b)

    def generate_data_graph(self, topn_links, risk2info=None):
        if risk2info is None:
            risk2info = self.__risk2info

        json_data = list()
        for risk_i in topn_links.keys():
            risk_info = risk2info[risk_i]
            category = risk_info['category']
            category_idx = risk_info['category_idx']
            risk_idx = risk_info['risk_idx']
            name = risk_info['label']
            info = risk_info
            links = []
            for risk_j, wij in topn_links[risk_i]:
                links.append(("{}.{}".format(risk2info[risk_j]['category'], risk2info[risk_j]['label']), wij))

            risk_record = {
                'group': category_idx,
                'id': risk_idx,
                'links': links,
                'name': "{}.{}".format(category, name),
                'info': info
            }

            json_data.append(risk_record)

        return json_data

    def extract_authors(self, rinfos):
        authors = set()
        for article in [x[0] for x in rinfos]:
            authors.update(set([a.lower() for a in article.get_dict()['authors']]))
        return authors

    def compute_tfidf(self, risk_i, risk_j, risk2entities, global_entity_freq):
        entity_tfidf = Counter()
        entities_i = set(risk2entities[risk_i])
        entities_j = set(risk2entities[risk_j])
        common_entities = entities_i.intersection(entities_j)
        entities_freq = Counter({k: v for k, v in risk2entities[risk_i].items() if k in common_entities}) + \
                        Counter({k: v for k, v in risk2entities[risk_j].items() if k in common_entities})
        for k in entities_freq.keys():
            # tf*idf computation.
            # Recommended tf–idf weighting scheme number 3: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
            entity_tfidf[k] = (1.0 + np.log1p(entities_freq[k])) * np.log1p(N / global_entity_freq[k])
        return entity_tfidf

    def rank_per_entity_type(self, entity_tfidf, e_types=['PERSON', 'ORG', 'GPE'], topn=10):
        rank = defaultdict(list)
        for ((e, e_type), _) in entity_tfidf.most_common():
            if e_type not in e_types:
                continue
            rank[e_type].append(e)
        # trim:
        for k in rank.keys():
            rank[k] = rank[k][0:topn]
        return dict(rank)

    def compute_entity_rank(self, topn_links, risk2entities, global_entity_freq):
        entity_rank = dict()
        for ri in topn_links.keys():
            if ri not in entity_rank:
                entity_rank[ri] = list()
            for (rj, wij) in topn_links[ri]:
                entity_tfidf = self.compute_tfidf(ri, rj, risk2entities, global_entity_freq)
                rank_e = self.rank_per_entity_type(entity_tfidf)
                entity_rank[ri].append((rj, wij, rank_e))
        return entity_rank

    def compute_entity_rank_for_timeseries(self, time2topn_links, time2risk_entities, global_entity_freq):
        entity_rank = dict()
        for t in time2topn_links.keys():
            if t not in entity_rank:
                entity_rank[t] = dict()
            entity_rank[t] = self.compute_entity_rank(time2topn_links[t], time2risk_entities[t], global_entity_freq)
        return entity_rank

    def generate_data_graph_with_entity(self, topn_links, risk2info=None):
        if risk2info is None:
            risk2info = self.__risk2info

        links = list()
        nodes_with_links = set()
        link_counter = 0
        for risk_i in topn_links.keys():
            nodes_with_links.add(risk_i)
            for risk_j, wij, entities in topn_links[risk_i]:
                nodes_with_links.add(risk_j)
                links.append(
                    {"id": link_counter, "source": risk_i, "target": risk_j, "weight": wij, "entities": entities})
                link_counter += 1
        nodes = {k: v for k, v in risk2info.items() if k in nodes_with_links}
        return {"nodes": nodes, "links": links}

    def generate_time2data_graph_with_entity(self, time2topn_links, risk2info=None):
        if risk2info is None:
            risk2info = self.__risk2info

        time2datagraph = dict()
        for t in time2topn_links.keys():
            topn_links = time2topn_links[t]
            graph_data_x = self.generate_data_graph_with_entity(topn_links, risk2info)
            time2datagraph[t] = graph_data_x
        return time2datagraph

    def extract_timeseries(self, links):
        time_keys = sorted(links.keys())
        xlinks = dict()
        for k in time_keys:
            kx = '{}W{:02}'.format(k[0], k[1])
            if k not in xlinks:
                xlinks[kx] = dict()
            xlinks[kx] = {'{}_{}'.format(ri, rj): wij for ri, rj, wij in links[k]}
        # the timeseries:
        xlinks = pd.DataFrame(xlinks)
        xlinks = xlinks.reindex(columns=sorted(xlinks.columns))
        xlinks = xlinks.T
        xlinks = xlinks.fillna(xlinks.median())
        return xlinks

    def aggregate_latest_entities(self, time2topn_links_and_entity_rank, window_size=4, topn=10):
        risk2risk_entities = dict()
        if len(time2topn_links_and_entity_rank) > window_size:
            lastest_entity_times = sorted(list(time2topn_links_and_entity_rank.keys()))[-window_size::]
        else:
            lastest_entity_times = sorted(list(time2topn_links_and_entity_rank.keys()))
        for t in lastest_entity_times:
            links = time2topn_links_and_entity_rank[t]
            for risk_i in links.keys():
                if risk_i not in risk2risk_entities:
                    risk2risk_entities[risk_i] = dict()
                for risk_j, w_ij, entities_ij in links[risk_i]:
                    if risk_j not in risk2risk_entities[risk_i]:
                        risk2risk_entities[risk_i][risk_j] = dict()
                    for entity_type in entities_ij.keys():
                        if entity_type not in risk2risk_entities[risk_i][risk_j]:
                            risk2risk_entities[risk_i][risk_j][entity_type] = Counter()
                        risk2risk_entities[risk_i][risk_j][entity_type].update(
                            {entity: 1.0 / (rank) for rank, entity in enumerate(entities_ij[entity_type], start=1)}
                        )

        # trim to topn entities
        for risk_i in risk2risk_entities.keys():
            for risk_j in risk2risk_entities[risk_i].keys():
                for entity_type in risk2risk_entities[risk_i][risk_j].keys():
                    risk2risk_entities[risk_i][risk_j][entity_type] = [x[0] for x in risk2risk_entities[risk_i][risk_j][
                        entity_type].most_common(topn)]
        return risk2risk_entities

    # add entities to top links:
    def add_entities_to_links(self, topn_links, risk2risk_entities):
        topn_links_with_entities = dict()
        for risk_i in topn_links.keys():
            for risk_j, w_ij in topn_links[risk_i]:
                # if we do not have entities to add for the link, we continue,
                # otherwise we would not have an explainable score, which might
                # still be OK if explainability is not required as in our case.
                if risk_i not in risk2risk_entities:
                    continue
                if risk_j not in risk2risk_entities[risk_i]:
                    continue
                # initialize:
                if risk_i not in topn_links_with_entities:
                    topn_links_with_entities[risk_i] = list()
                topn_links_with_entities[risk_i].append((risk_j, w_ij, risk2risk_entities[risk_i][risk_j]))
        return topn_links_with_entities

    def load(self, fname):
        o = None
        with open(fname, "rb") as fin:
            o = pickle.load(fin)
        return o

    def dump(self, o, fname):
        with open(fname, "wb") as fout:
            pickle.dump(o, fout)
