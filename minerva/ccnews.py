from __future__ import print_function, unicode_literals

"""
Provides functionality to crawl and extract news articles from a single local gzipped WARC file from commoncrawl.org.
Filter criteria, such as publish date and host list, can be defined.
"""

import logging
import os
import time

from ago import human
from dateutil import parser

from scrapy.utils.log import configure_logging
from six.moves import urllib
from warcio.archiveiterator import ArchiveIterator

from newsplease import NewsPlease, NewscrawlerItem

from newsplease.pipeline.extractor.article_extractor import Extractor
from newsplease.pipeline.pipelines import ExtractedInformationStorage
from dotmap import DotMap

from bs4 import UnicodeDammit

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


__author__ = "Libre AI"
__copyright__ = "Copyright 2018"
__credits__ = ["newsplease"]


class CommonCrawlExtractor:
    # hosts (if None or empty list, any host is OK)
    __filter_valid_hosts = []  # example: ['elrancaguino.cl']
    # start date (if None, any date is OK as start date), as datetime
    __filter_start_date = None
    # end date (if None, any date is OK as end date)
    __filter_end_date = None
    # if date filtering is string, e.g., if we could not detect the date of an article, we will discard the article
    __filter_strict_date = True
    # continue after error
    __continue_after_error = False
    # log level
    __log_level = logging.INFO
    # whether to delete file or not after extraction
    __delete_warc_after_extraction = True
    # event handler called when an article was extracted successfully and passed all filter criteria
    __callback_on_article_extracted = None
    # logging
    logging.basicConfig(level=__log_level)
    __logger = logging.getLogger(__name__)

    def __setup(self):
        """
        Setup
        :return:
        """

        # make loggers quite
        configure_logging({"LOG_LEVEL": "ERROR"})
        logging.getLogger('requests').setLevel(logging.CRITICAL)
        logging.getLogger('readability').setLevel(logging.CRITICAL)
        logging.getLogger('PIL').setLevel(logging.CRITICAL)
        logging.getLogger('newspaper').setLevel(logging.CRITICAL)
        logging.getLogger('newsplease').setLevel(logging.CRITICAL)
        logging.getLogger('bs4').setLevel(logging.CRITICAL)
        logging.getLogger('chardet').setLevel(logging.CRITICAL)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)

        # set own logger
        logging.basicConfig(level=self.__log_level)
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(self.__log_level)


    def __filter_by_valid_hosts(self, warc_record, article=None):
        pass_filter = True  # pass by default if valid_hosts are not specified
        # filter by host
        if self.__filter_valid_hosts:
            # extract source domain
            url = warc_record.rec_headers.get_header('WARC-Target-URI')
            source_domain = urllib.parse.urlparse(url).hostname if url != '' else ''

            # check if the source_domain is in our list of valid hosts:
            for valid_host in self.__filter_valid_hosts:
                if source_domain.endswith(valid_host):
                    pass_filter = True
                    break
            else:
                pass_filter = False
        return pass_filter, article

    def __filter_by_date(self, warc_record, article=None):
        pass_filter = True
        # filter by date
        if self.__filter_start_date or self.__filter_end_date:
            if not article:
                article = CommonCrawlExtractor.from_warc(warc_record)

            publishing_date = self.__get_publishing_date(warc_record, article)
            if not publishing_date:
                if self.__filter_strict_date:
                    pass_filter = False
            else:  # here we for sure have a date
                # is article published too early?
                if self.__filter_start_date and publishing_date < self.__filter_start_date:
                    pass_filter = False

                if self.__filter_end_date and publishing_date > self.__filter_end_date:
                    pass_filter = False

        return pass_filter, article

    def __filter_record(self, warc_record, filters, article=None):
        """
        Returns true if a record passes all tests: hosts, publishing date
        :param warc_record:
        :return: A tuple of (True or False) and an article (might be None)
        """
        pass_filter = True

        for f in filters:
            pass_filter, article = f(warc_record, article)
            if pass_filter is False:
                break

        return pass_filter, article

    def __get_publishing_date(self, warc_record, article):
        """
        Extracts the publishing date from the record
        :param warc_record:
        :return:
        """
        if 'publish_date' in article:
            return parser.parse(article.publish_date)
        else:
            return None

    def __process_warc_gz_file(self, path_name):
        """
        Iterates all transactions in one WARC file and for each transaction tries to extract an article object.
        Afterwards, each article is checked against the filter criteria and if all are passed, the function
        on_valid_article_extracted is invoked with the article object.
        :param path_name:
        :return:
        """
        counter_article_total = 0
        counter_article_passed = 0
        counter_article_discarded = 0
        start_time = time.time()

        articles = []

        with open(path_name, 'rb') as stream:
            for record in ArchiveIterator(stream):
                try:
                    if record.rec_type == 'response':
                        counter_article_total += 1

                        # if the article passes filter tests, we notify the user
                        filter_pass, article = self.__filter_record(record,
                                                                    filters=[self.__filter_by_valid_hosts,
                                                                             self.__filter_by_date]
                                                                    )
                        if filter_pass:
                            counter_article_passed += 1

                            if not article:
                                article = CommonCrawlExtractor.from_warc(record)

                            self.__logger.debug('article pass (%s; %s; %s)', article.source_domain, article.date_publish,
                                               article.title)
                            # collecting the candidate warc records
                            articles.append(article)
                        else:
                            counter_article_discarded += 1

                            if article:
                                self.__logger.debug('article discard (%s; %s; %s)', article.source_domain,
                                                   article.date_publish,
                                                   article.title)
                            else:
                                self.__logger.debug('article discard (%s)',
                                                   record.rec_headers.get_header('WARC-Target-URI'))

                        if counter_article_total % 10 == 0:
                            elapsed_secs = time.time() - start_time
                            secs_per_article = elapsed_secs / counter_article_total
                            self.__logger.info('statistics')
                            self.__logger.info('pass = %i, discard = %i, total = %i', counter_article_passed,
                                               counter_article_discarded, counter_article_total)
                            self.__logger.info('extraction from {} WARC file started %s; %f s/article'.format(path_name),
                                               human(start_time), secs_per_article)
                except Exception as e:
                    self.__logger.exception(e)
                    continue

        # cleanup
        if self.__delete_warc_after_extraction:
            os.remove(path_name)

        return articles

    def __process_warc_gz_file_with_callback(self, path_name):
        """
        Iterates all transactions in one WARC file and for each transaction tries to extract an article object.
        Afterwards, each article is checked against the filter criteria and if all are passed, the function
        on_valid_article_extracted is invoked with the article object.
        :param path_name:
        :return:
        """
        counter_article_total = 0
        counter_article_passed = 0
        counter_article_discarded = 0
        start_time = time.time()

        with open(path_name, 'rb') as stream:
            for record in ArchiveIterator(stream):
                try:
                    if record.rec_type == 'response':
                        counter_article_total += 1

                        # if the article passes filter tests, we notify the user
                        filter_pass, article = self.__filter_record(record,
                                                                    filters=[self.__filter_by_valid_hosts, self.__filter_by_date]
                                                                    )
                        if filter_pass:
                            counter_article_passed += 1

                            if not article:
                                article = CommonCrawlExtractor.from_warc(record)

                            self.__logger.info('article pass (%s; %s; %s)', article.source_domain, article.date_publish,
                                               article.title)
                            # article extracted fine, we execute the callback:
                            self.__callback_on_article_extracted(article)
                        else:
                            counter_article_discarded += 1

                            if article:
                                self.__logger.info('article discard (%s; %s; %s)', article.source_domain,
                                                   article.date_publish,
                                                   article.title)
                            else:
                                self.__logger.info('article discard (%s)',
                                                   record.rec_headers.get_header('WARC-Target-URI'))

                        if counter_article_total % 10 == 0:
                            elapsed_secs = time.time() - start_time
                            secs_per_article = elapsed_secs / counter_article_total
                            self.__logger.info('statistics')
                            self.__logger.info('pass = %i, discard = %i, total = %i', counter_article_passed,
                                               counter_article_discarded, counter_article_total)
                            self.__logger.info('extraction from current WARC file started %s; %f s/article',
                                               human(start_time), secs_per_article)
                except Exception as e:
                    self.__logger.exception(e)
                    continue

        # cleanup
        if self.__delete_warc_after_extraction:
            os.remove(path_name)

    def __run_local_extraction(self, cc_gzip_file):
        """
        Main execution method, which consists of: get an up-to-date list of WARC files, and for each of them: download
        and extract articles. Each article is checked against a filter. Finally, for each valid article the method
        on_valid_article_extracted will be invoked after the extraction of the article has completed.
        :return:
        """
        self.__setup()
        return self.__process_warc_gz_file(cc_gzip_file)

    def extract_from_commoncrawl_gzip_local(self, cc_gzip_file, valid_hosts=None,
                                          start_date=None, end_date=None, strict_date=True, continue_after_error=True,
                                          log_level=logging.INFO, delete_warc_after_extraction=False):
        """
        Crawl and extract articles form the news crawl provided by commoncrawl.org. For each article that was extracted
        successfully the callback function callback_on_article_extracted is invoked where the first parameter is the
        article object.
        """

        self.__filter_valid_hosts = valid_hosts
        self.__filter_start_date = start_date
        self.__filter_end_date = end_date
        self.__filter_strict_date = strict_date

        self.__continue_after_error = continue_after_error
        self.__log_level = log_level
        self.__delete_warc_after_extraction = delete_warc_after_extraction

        return self.__run_local_extraction(cc_gzip_file)

    # def extract_from_commoncrawl_gzip_local(self, cc_gzip_file, callback_on_article_extracted, valid_hosts=None,
    #                                       start_date=None, end_date=None, strict_date=True, continue_after_error=True,
    #                                       log_level=logging.ERROR, delete_warc_after_extraction=False):
    #     """
    #     Crawl and extract articles form the news crawl provided by commoncrawl.org. For each article that was extracted
    #     successfully the callback function callback_on_article_extracted is invoked where the first parameter is the
    #     article object.
    #     """
    #
    #     self.__filter_valid_hosts = valid_hosts
    #     self.__filter_start_date = start_date
    #     self.__filter_end_date = end_date
    #     self.__filter_strict_date = strict_date
    #
    #     self.__continue_after_error = continue_after_error
    #     self.__callback_on_article_extracted = callback_on_article_extracted
    #     self.__log_level = log_level
    #     self.__delete_warc_after_extraction = delete_warc_after_extraction
    #
    #     self.__run_local_extraction(cc_gzip_file)

    @staticmethod
    def from_html(html, url=None, download_date=None):
        """
        Extracts relevant information from an HTML page given as a string. This function does not invoke scrapy but only
        uses the article extractor. If you have the original URL make sure to provide it as this helps NewsPlease
        to extract the publishing date and title.
        :param html:
        :param url:
        :return:
        """

        # extractors_pipeline = ['newspaper_extractor', 'readability_extractor', 'date_extractor',
        # 'lang_detect_extractor']

        # do not extract lang here, we will do it later at the level of the nlp processing
        extractors_pipeline = ['newspaper_extractor', 'readability_extractor', 'date_extractor']

        extractor = Extractor(extractors_pipeline)

        title_encoded = ''.encode()
        if not url:
            url = ''

        # if an url was given, we can use that as the filename
        filename = urllib.parse.quote_plus(url) + '.json'

        item = NewscrawlerItem()
        item['spider_response'] = DotMap()
        item['spider_response'].body = html
        item['url'] = url
        item['source_domain'] = urllib.parse.urlparse(url).hostname.encode() if url != '' else ''.encode()
        item['html_title'] = title_encoded
        item['rss_title'] = title_encoded
        item['local_path'] = None
        item['filename'] = filename
        item['download_date'] = download_date
        item['modified_date'] = None
        item = extractor.extract(item)

        tmp_article = ExtractedInformationStorage.extract_relevant_info(item)
        final_article = ExtractedInformationStorage.convert_to_class(tmp_article)
        return final_article

    @staticmethod
    def from_warc(warc_record):
        """
        Extracts relevant information from a WARC record. This function does not invoke scrapy but only uses the article
        extractor.
        :return:
        """
        # html = str(warc_record.raw_stream.read())
        html = UnicodeDammit(warc_record.raw_stream.read()).unicode_markup
        #html = UnicodeDammit(warc_record.content_stream().read()).unicode_markup
        url = warc_record.rec_headers.get_header('WARC-Target-URI')
        download_date = warc_record.rec_headers.get_header('WARC-Date')
        article = CommonCrawlExtractor.from_html(html, url=url, download_date=download_date)
        return article