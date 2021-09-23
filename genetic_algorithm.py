import csv
import logging
import math
import os
import pickle as pkl
import random
import string
import sys
from operator import itemgetter
from collections import Counter
from typing import List

import morfeusz2
import numpy as np
from stempel import StempelStemmer

logger = logging.getLogger('info_logs')
log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s", '%H:%M:%S')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logging.basicConfig(filename='genetic_algorithm_logs.log',
                    filemode='a',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


class GeneticAlgorithm(object):

    def __init__(self, pop_size=100, gen=100, p_x=0.7, p_m=0.1, tour=5, use_tour=False,
                 use_pmx_crossover=False, use_elitist_selection=False, use_mutation=True, use_inversion=False,
                 triple_mwe=False, get_acc=False):
        self.pop_size = pop_size
        self.gen = gen
        self.p_x = p_x
        self.p_m = p_m
        self.n = 9  # number of measures to optimize
        self.mwe_count = 100000  # 10556507  # number of mwe to be found
        self.tour = tour
        self.use_tour = use_tour
        self.use_pmx_crossover = use_pmx_crossover
        self.use_elitist_selection = use_elitist_selection
        self.best_chromosome = np.array([])
        self.use_mutation = use_mutation
        self.use_inversion = use_inversion
        self.counter = 0
        self.cur_pop = None
        self.evaluated_pop = None
        self.selected_pop = None
        self.min_fitness = math.inf
        self.max_fitness = -math.inf
        self.evaluation_difference_sum = 0
        self.new_pop = None
        self.sum_of_probabilities = 0.0
        self.pop_probabilities = np.empty(gen, dtype=np.float64)
        self.worst_history = np.empty(gen, dtype=np.float64)
        self.avg_history = np.empty(gen, dtype=np.float64)
        self.best_history = np.empty(gen, dtype=np.float64)
        self.measure_dict = {}
        self.wordnet_results = []
        self.wordnet_mwe = []
        # self.lemmatizer = StempelStemmer.polimorf()
        self.morf = morfeusz2.Morfeusz()  # initialize Morfeusz object
        self.triple_mwe = triple_mwe
        self.get_acc = get_acc
        pass

    def format_mwe(self, mwe: string) -> string:
        mwe_words: List = mwe.split(' ')

        # cleaning, because the file format is not very good
        while '' in mwe_words:
            mwe_words.remove('')

        for i, mwe_word in enumerate(mwe_words):
            mwe_words[i] = mwe_words[i].replace('\xad', '')  # remove \xad
            mwe_words[i] = mwe_words[i].replace(
                '\xc2', '')  # remove soft hyphen
            mwe_words[i] = mwe_words[i].replace(
                '\x99', '')  # remove soft hyphen
            mwe_words[i] = mwe_words[i].replace(
                '\x9c', '')  # remove soft hyphen
            mwe_words[i] = mwe_words[i].replace(
                '\x82', '')  # remove soft hyphen

        return ' '.join(mwe_words)

    def check_type_correctness(self, mwe_type) -> bool:
        mwe_types = mwe_type.split(' + ')

        type_correctness = True

        # check if MWE consists of more than 2 words
        if not self.triple_mwe and len(mwe_types) != 2:
            type_correctness = False

        # check if there is 'nieciągłość" in MWE types
        if 'nieciągłość' in mwe_types:
            type_correctness = False

        return type_correctness

    def check_mwe_correctness(self, morf, mwe: string) -> bool:
        ign = False  # flag informing that the mwe is incorrect
        mwe_words: List = mwe.split(' ')

        while '' in mwe_words:
            mwe_words.remove('')

        i = len(mwe_words) - 1
        while not ign and i >= 0:
            mwe_word = mwe_words[i]
            if 'http' in mwe_word or 'www' in mwe_word or '.net' in mwe_word or '.pl' in mwe_word:  # detect incorrect words
                ign = True
                continue
            ana = morf.analyse(mwe_word)
            ign: bool = ana[0][2][2] == 'ign'
            i -= 1

        return not ign

    def load_measure_results(self):
        logger.info(f'Loading measure results')
        dir_path = os.path.join(
            '/', 'data4', 'netherwulf', 'mewex', 'docker', 'train_results')

        file_paths = ['result_train_frequency_biased_mutual_dependency.txt',
                      'result_train_inversed_expected_frequency.txt',
                      'result_train_mutual_dependency.txt',
                      'result_train_pearsons_chi2.txt',
                      'result_train_pointwise_mutual_information.txt',
                      'result_train_specific_exponential_correlation.txt',
                      'result_train_w_pearsons_chi2.txt',
                      'result_train_w_specific_exponential_correlation.txt',
                      'result_train_zscore.txt']

        mwe_dict = {}

        for file_path_num, file_path in enumerate(file_paths):
            measure_name = '_'.join(file_path.split('.')[0].split('_')[2:])
            logger.info(f'Loading results for measure: {measure_name}')
            mwe_dict[measure_name] = []

            with open(os.path.join(dir_path, file_path), 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                for i, row in enumerate(csv_reader):
                    if i == 0 or i == 1:
                        continue

                    measure_value = row[0]
                    count = int(float(row[1]))
                    collocation_type = row[2]
                    mwe = row[4].split(':')[1].split('(')[0].strip()

                    if not self.check_type_correctness(collocation_type):
                        continue

                    # clean MWE
                    formatted_mwe = self.format_mwe(mwe)

                    # check if each word is in the morfeusz dictionary
                    if not self.check_mwe_correctness(self.morf, formatted_mwe):
                        continue

                    # lemmatized_mwe = ' '.join(
                    #     [str(self.lemmatizer.stem(word.lower())) if word not in string.punctuation else word for word in
                    #      mwe.split(' ')])
                    lemmatized_mwe = ' '.join(
                        [str(self.morf.analyse(word.lower())[0][2][1]) if word not in string.punctuation else word for word in
                         formatted_mwe.split(' ')])
                    # print(f'MWE: {repr(mwe)}',
                    #       f'formatted MWE: {repr(formatted_mwe)}',
                    #       f'lemmatized MWE: {repr(lemmatized_mwe)}',
                    #       sep='\n')

                    # mwe_dict[measure_name].extend([(measure_value, lemmatized_mwe) for _ in range(count)])
                    mwe_dict[measure_name].extend(
                        [(measure_value, lemmatized_mwe)])

                    if len(mwe_dict[measure_name]) >= self.mwe_count:
                        mwe_dict[measure_name] = mwe_dict[measure_name][:self.mwe_count]
                        break

        return mwe_dict

    # read MWEs from Słowosieć tsv file
    def read_mwe(self, filepath: str) -> List[str]:
        with open(filepath, "r", encoding="utf-8") as f:
            content = list(csv.reader(f, delimiter="\t"))
            mwe_list = [sublist[2] for sublist in content[1:]]

            for i, mwe in enumerate(mwe_list):
                print(f'mwe_list[i]: {mwe_list[i]}\nmwe: {mwe}')
                mwe = ' '.join([word for word in mwe.split(' ') if word != ''])
                mwe_list[i] = ' '.join(
                    [str(self.morf.analyse(word.lower())[0][2][1]) for word in mwe.split(' ')])

            return mwe_list

    def load_wordnet_results(self):
        logger.info(f'Loading true MWE list')
        with open('train_found_mwe.pkl', 'rb') as f:
            wordnet_results = pkl.load(f)

        return wordnet_results

    def initialize(self):
        self.measure_dict = self.load_measure_results()
        self.wordnet_results = self.load_wordnet_results()
        self.wordnet_mwe = self.read_mwe('mwe.tsv')
        logger.info(f'Measure-based and true MWEs loaded')
        # self.cur_pop = np.array([np.array([round(random.random(), 3) for _ in range(self.n)]) for _ in range(self.pop_size)])
        self.cur_pop = np.array(
            [np.array([round(random.uniform(0, 50), 3) for _ in range(self.n)]) for _ in range(self.pop_size)])
        # [np.array([round(random.uniform(0, 1e16), 3) for _ in range(self.n)]) for _ in range(self.pop_size)])

    def get_measure_mwe(self, measure_name, weight):
        """ Calculate values of measure based on weight and return sorted list of tuples (value, mwe) """
        mwe_list = self.measure_dict[measure_name].copy()
        max_measure_value = max([float(mwe_tuple[0])
                                for mwe_tuple in mwe_list])
        # mwe_list = [((float(mwe_tuple[0]) * weight), mwe_tuple[1])
        #             for mwe_tuple in mwe_list]
        mwe_list = [((abs(float(mwe_tuple[0]) / max_measure_value) * 50 * weight), mwe_tuple[1])
                    for mwe_tuple in mwe_list]
        # mwe_list = [(abs((float(mwe_tuple[0]) % 30) * weight), mwe_tuple[1]) for mwe_tuple in mwe_list]

        return mwe_list

    @staticmethod
    def get_f1_score(tp: int, fp: int, fn: int) -> float:
        # calculate precision, recall and f1-score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        return f1_score

    @staticmethod
    def get_accuracy(tp: int, fp: int, fn: int) -> float:
        # calculate accuracy
        accuracy = tp / (tp + fp + fn)

        return accuracy

    def get_chromosome_fitness(self, mwe_dict, chromosome, get_acc=False) -> float:
        """ Calculate F1 measure for sorted lists of tuples (value, mwe) """
        mwe_list = []

        for measure_name in sorted(mwe_dict.keys()):
            mwe_list.extend(mwe_dict[measure_name])

        sorted_mwe_list = sorted(mwe_list, key=itemgetter(0), reverse=True)
        sorted_mwe_list = sorted_mwe_list[:self.mwe_count]
        sorted_mwe_list = [mwe_tuple[1] for mwe_tuple in sorted_mwe_list]
        # true_mwe_list = self.wordnet_results.copy()
        true_mwe_list = self.wordnet_mwe.copy()

        # intersection of sorted_mwe_list and true_mwe_list - true mwe that were found
        sorted_mwe_list_temp = sorted_mwe_list.copy()
        true_mwe_list_temp = true_mwe_list.copy()
        # intersection = Counter(
        #     sorted_mwe_list_temp) & Counter(true_mwe_list_temp)
        intersection = set(sorted_mwe_list_temp).intersection(
            set(true_mwe_list_temp))
        # tp = len(list(intersection.elements()))
        tp = len(intersection)

        # difference between sorted_mwe_list and true_mwe_list - mwe that shouldn't be detected, but were
        sorted_mwe_list_temp = sorted_mwe_list.copy()
        true_mwe_list_temp = true_mwe_list.copy()
        # fp_list = Counter(sorted_mwe_list_temp) - Counter(true_mwe_list_temp)
        fp_list = set(sorted_mwe_list_temp) - set(true_mwe_list_temp)
        # fp = len(list(fp_list))
        fp = len(fp_list)

        # difference between true_mwe_list and sorted_mwe_list - mwe that should be detected, but weren't
        sorted_mwe_list_temp = sorted_mwe_list.copy()
        true_mwe_list_temp = true_mwe_list.copy()
        # fn_list = Counter(true_mwe_list_temp) - Counter(sorted_mwe_list_temp)
        fn_list = set(true_mwe_list_temp) - set(sorted_mwe_list_temp)
        # fn = len(list(fn_list))
        fn = len(fn_list)

        # calculate fitness
        if get_acc:
            fitness = self.get_accuracy(tp, fp, fn)
        else:
            fitness = self.get_f1_score(tp, fp, fn)

        if fitness > self.max_fitness:
            logger.info(f'New best fitness: {fitness}')
            logger.info(f'New best chromosome: {chromosome}')
            self.max_fitness = fitness
            self.best_chromosome = chromosome

        return fitness

    def evaluate(self, chromosome) -> float:
        mwe_dict = {}
        for measure_id, measure_name in enumerate(sorted(self.measure_dict.keys())):
            mwe_dict[measure_name] = self.get_measure_mwe(
                measure_name, chromosome[measure_id])

        # optimization based on accuracy
        return self.get_chromosome_fitness(mwe_dict, chromosome, get_acc=self.get_acc)

    def correct_genes(self):
        for chromosome_id, chromosome in enumerate(self.cur_pop):
            for gene_id, gene in enumerate(chromosome):
                if (gene_id == 0 or gene_id == 5) and gene > 0:
                    self.cur_pop[chromosome_id][gene_id] = -1.0 * gene

                if gene_id in [1, 3, 6] and gene > 1.0:
                    self.cur_pop[chromosome_id][gene_id] = gene % 1e-14

                if gene_id == 8 and gene > 1.0:
                    self.cur_pop[chromosome_id][gene_id] = gene % 1e-6

                if gene_id in [2, 4, 5, 7] and gene < 2.0:
                    self.cur_pop[chromosome_id][gene_id] = round(
                        random.uniform(0, 100), 3)

    def evaluation(self, gen_num):
        # self.correct_genes()
        self.evaluated_pop = np.array([self.evaluate(i) for i in self.cur_pop])
        self.max_fitness = np.amax(self.evaluated_pop) if np.amax(
            self.evaluated_pop) > self.max_fitness else self.max_fitness
        logger.info(f'Current best fitness: {self.max_fitness}')
        logger.info(
            f'Current best chromosome: {self.cur_pop[np.argmax(self.evaluated_pop)]}')

        self.min_fitness = np.amin(self.evaluated_pop) if np.amin(
            self.evaluated_pop) < self.min_fitness else self.min_fitness
        self.evaluation_difference_sum = sum(
            self.max_fitness - self.evaluated_pop)
        self.worst_history[gen_num] = np.amin(self.evaluated_pop)
        self.avg_history[gen_num] = np.average(self.evaluated_pop)
        self.best_history[gen_num] = np.amax(self.evaluated_pop)

    def roulette_prob(self, cur_fitness):

        return (self.max_fitness - cur_fitness) / self.evaluation_difference_sum

    def selection(self):
        self.selected_pop = np.empty((self.pop_size, self.n), float)
        # self.best_chromosome = self.cur_pop[np.argmax(self.evaluated_pop)]
        # if self.use_elitist_selection:
        # self.best_chromosome = self.cur_pop[np.argmax(self.evaluated_pop)]
        if self.use_tour:
            for j in range(self.pop_size):
                tour_members = np.empty((0, self.n), float)
                tour_members_ids = random.sample(
                    range(0, self.pop_size), self.tour)
                for i in tour_members_ids:
                    chromosome = self.cur_pop[i]
                    tour_members = np.append(
                        tour_members, np.array([chromosome]), axis=0)
                evaluated_tour_members = np.array(
                    [self.evaluated_pop[tm_id] for tm_id in tour_members_ids])
                self.selected_pop[j] = tour_members[np.argmax(
                    evaluated_tour_members)]
        else:
            # self.pop_probabilities = np.array([])
            self.sum_of_probabilities = 0.0
            for eval_chrom_id, j in enumerate(self.evaluated_pop):
                probability = self.sum_of_probabilities + self.roulette_prob(j)
                self.pop_probabilities[eval_chrom_id] = probability
                self.sum_of_probabilities += self.roulette_prob(j)
            for i, obj in enumerate(self.pop_probabilities):
                if obj >= random.random():
                    self.selected_pop = np.append(
                        self.selected_pop, np.array([self.cur_pop[i]]), axis=0)

    def ox_crossover(self, parent_a, parent_b):
        a = random.randint(1, len(parent_a) - 3)
        b = random.randint(a + 1, len(parent_a) - 2)

        child_a = np.zeros(self.n, float)
        child_b = np.zeros(self.n, float)

        # insert fragment of the other parent
        for i in range(a, b + 1):
            np.put(np.asarray(child_a), i, parent_b[i])
            np.put(np.asarray(child_b), i, parent_a[i])

        # repair missing genes
        for j in range(len(parent_a) - (b - a)):
            repairing_index = b + j + 1
            if repairing_index > (len(parent_a) - 1):
                repairing_index = (repairing_index % len(parent_a))

            # repair genes of child_a
            for k in range(len(parent_a)):
                parent_index = repairing_index + k
                if parent_index > (len(parent_a) - 1):
                    parent_index = (parent_index % len(parent_a))
                if not np.asarray(parent_a)[parent_index] in np.asarray(child_a):
                    np.put(child_a, repairing_index, parent_a[parent_index])
                    break

            # repair genes of child_b
            for l in range(len(parent_b)):
                parent_index = repairing_index + l
                if parent_index > (len(parent_b) - 1):
                    parent_index = (parent_index % len(parent_b))
                if not np.asarray(parent_b)[parent_index] in child_b:
                    np.put(child_b, repairing_index, parent_b[parent_index])
                    break
        return child_a, child_b

    def pmx_crossover(self, parent_a, parent_b):
        a = random.randint(1, len(parent_a) - 3)
        b = random.randint(a + 1, len(parent_a) - 2)

        child_a = np.zeros(self.n, float)
        child_b = np.zeros(self.n, float)

        # insert fragment of another parent genotype
        for i in range(a, b + 1):
            np.put(np.asarray(child_a), i, parent_b[i])
            np.put(np.asarray(child_b), i, parent_a[i])

        # repair genes
        for j in range(len(parent_a) - (b - a)):
            repairing_index = b + j + 1
            if repairing_index > (len(parent_a) - 1):
                repairing_index = (repairing_index % len(parent_a)) - 1

            # repair genes of child_a
            for k in range(len(parent_a)):
                parent_index = repairing_index + k
                if parent_index > (len(parent_a) - 1):
                    parent_index = (parent_index % len(parent_a))
                if not np.asarray(parent_a)[parent_index] in np.asarray(child_a):
                    np.put(child_a, repairing_index, parent_a[parent_index])
                    break
                # map genes in child_a to the ones at the same position in the child_B
                else:
                    gene = np.asarray(parent_a)[parent_index]
                    genes_checked = 0
                    while gene in np.asarray(child_a) and genes_checked <= (b - a):
                        gene = child_b[np.where(child_a == gene)[0]]
                        genes_checked += 1
                    if gene not in np.asarray(child_a):
                        np.put(child_a, repairing_index, gene)
                        break

            # repair genes of child_b
            for l in range(len(parent_b)):
                parent_index = repairing_index + l
                if parent_index > (len(parent_b) - 1):
                    parent_index = (parent_index % len(parent_b))
                if not np.asarray(parent_b)[parent_index] in child_b:
                    np.put(child_b, repairing_index, parent_b[parent_index])
                    break
                # map genes in child_a to the ones at the same position in the child_B
                else:
                    gene = np.asarray(parent_b)[parent_index]
                    genes_checked = 0
                    while gene in np.asarray(child_b) and genes_checked <= (b - a):
                        gene = child_a[np.where(child_b == gene)[0]]
                        genes_checked += 1
                    if gene not in np.asarray(child_b):
                        np.put(child_b, repairing_index, gene)
                        break

        return child_a, child_b

    def crossover(self):
        self.new_pop = np.array([])
        if self.use_elitist_selection:
            self.new_pop = np.append(
                self.new_pop, self.best_chromosome, axis=0)
        parent_1 = None
        for i in self.selected_pop:
            if np.random.random() <= self.p_x:
                if parent_1 is not None:
                    if self.use_pmx_crossover:
                        child_1, child_2 = self.pmx_crossover(parent_1, i)
                    else:
                        child_1, child_2 = self.ox_crossover(parent_1, i)
                    self.new_pop = np.append(self.new_pop, child_1, axis=0)
                    self.new_pop = np.append(self.new_pop, child_2, axis=0)
                    parent_1 = None
                else:
                    parent_1 = i
        self.new_pop = np.reshape(
            self.new_pop, [int(len(self.new_pop) / self.n), self.n])

        # repair missing samples in the population
        while len(self.new_pop) < self.pop_size:
            self.new_pop = np.append(self.new_pop,
                                     np.array(
                                         [self.cur_pop[np.random.randint(0, self.pop_size)]]),
                                     axis=0)

        # accept the newly created generation as the new one
        self.new_pop = self.new_pop.astype(float)
        self.cur_pop = np.copy(self.new_pop)
        self.new_pop = None

    def mutation(self):
        for chromosome in self.cur_pop:
            for i, gene in enumerate(chromosome):
                if np.random.rand() <= self.p_m:
                    chromosome[i] += round(random.uniform(-50.0, 50.0), 3)
                    # chromosome[i] += round(random.uniform(-0.25, 0.25), 3)
                    # chromosome[i] += round(random.uniform(-1e16, 1e16), 3)

    def inversion(self):
        inversion_beginning = random.randint(1, self.n - 3)
        inversion_ending = random.randint(inversion_beginning + 1, self.n - 2)
        for chromosome in self.cur_pop:
            if np.random.rand() <= self.p_m:
                chromosome[inversion_beginning:inversion_ending] = np.flip(
                    chromosome[inversion_beginning:inversion_ending])

    def run(self):
        best_fitnesses = np.array([])
        logger.info(f'Starting initialization')
        self.initialize()
        logger.info(f'Starting evaluation of initialized pop')
        self.evaluation(0)
        for i in range(self.gen):
            logger.info(f'Starting selection gen: {i}')
            self.selection()
            logger.info(f'Starting crossover gen: {i}')
            self.crossover()
            if self.use_mutation:
                self.mutation()
            if self.use_inversion:
                self.inversion()
            logger.info(f'Starting evaluation gen: {i}')
            self.evaluation(i)
            logger.info(
                f'Best fitness after generation: {str(self.max_fitness)}')
            logger.info(
                f'Best chromosome after generation: {self.best_chromosome}')
            best_fitnesses = np.append(best_fitnesses, self.max_fitness)
        logger.info(f'Average best fitness: {str(np.average(best_fitnesses))}')
        return np.average(best_fitnesses)

    def run_line_chart(self):
        self.initialize()
        self.evaluation(0)
        for i in range(self.gen):
            self.selection()
            self.crossover()
            if self.use_mutation:
                self.mutation()
            if self.use_inversion:
                self.inversion()
            self.evaluation(i)
        generations = np.arange(self.gen + 1)
        return self.worst_history, self.avg_history, self.best_history, generations
