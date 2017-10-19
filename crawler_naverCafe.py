# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import re
from collections import defaultdict
from selenium.webdriver.common.keys import Keys
import copy
import datetime
import pandas as pd
from collections import deque
import os
import psutil

# 시작 메모리 체크, 시간체크 import os, import psutil, import time 필요
proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]
start_time = time.time()
proc = psutil.Process(os.getpid())

class MenuIdReader():

    def __init__(self):
        self.menuid_dataframe = pd.read_csv('/users/kyungchankim/Documents/Spyder/Python_Study/PythonCrawler/' +'club_menu_id.csv')
        self.menuid_dataframe = self.menuid_dataframe.transpose()


    def menuIDPrint(self, club_name):
        for i in self.menuid_dataframe.ix[club_name]:
            for menuid in i.split(','):
                print(menuid)
        self.menuid = input('검색하실 게시판의 ID번호를 입력해주세요: ')
        return self.menuid

class NaverCafeCrawler(MenuIdReader):
    FILE_PATH = '/users/kyungchankim/Documents/Spyder/Python_Study/PythonCrawler/'
    CHROME_DRIVER_PATH = '/users/kyungchankim/Documents/Spyder/'
    CLUB_ID_DIC = { 'consumerizm':'22130984', 'dieselmania': '11262350', 'cosmania':'10050813', 'joonggonara':'10050146', 'specup': '15754634', 'sheiszzz':'15161846',
                    'honeymoondc':'10095818', 'feko':'10912875'}

    #cnt는 키워드 검색시 더보기 버튼 누르는 횟수 Pages는 전체 포스팅 크롤링 시 몇 페이지까지 할 것인가, keyword는 검색 키워드
    def __init__(self): # Pages=100 약 3000개 게시물 , cnt=100 검색결과 2000건
        self.clubname = None
        self.version = 'mobile'
        self.menuid = None
        self.isCommentCrwaling = 1
        #self.pre_article = deque(maxlen=1)
        self.sort = 1    #검색시 결과 값 정렬
        self.keyword = None
        self.more_cnt = 120
        self.pages = 1000
        self.choice_board = 2
        self.showSearchMenu()

        self.club_id = NaverCafeCrawler.CLUB_ID_DIC[self.clubname]
        self.favorite_board = False
        self.article_id_list = []
        self.error_article_id = []
        self.data_items = defaultdict(list)
        self.set_chrome_driver()
        self.playCrawling()

    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(NaverCafeCrawler.CHROME_DRIVER_PATH + 'chromedriver')

    # 네이버 로그인, url은 특정 카페에 있는 로그인 페이지, consumerizm, dieselmania
    def naverLogin(self):
        cafe_login_url = "https://nid.naver.com/nidlogin.login?svctype=262144&url=http%3A%2F%2Fm.cafe.naver.com%2FArticleAllList.nhn%3Fcluburl%3D{0}".format(self.clubname)
        self.driver.get(cafe_login_url)
        self.driver.find_element_by_id("id").send_keys("")
        self.driver.find_element_by_id("pw").send_keys("")
        self.driver.find_element_by_id("pw").submit()
        time.sleep(5)

    #즐겨찾는(특정) 게시판 전체보기
    # def selectFavoriteBoard(self):
    #     if self.favorite_board == True:
    #         self.driver.find_element_by_xpath("//a[contains(@onclick,'ctp.favtab')]").click()
    #         time.sleep(0.5)
    #         self.driver.find_element_by_xpath("//a[contains(@onclick,'fav.nmmore')]").click()
    #         time.sleep(0.5)

    # 메뉴ID별 게시판으로 이동
    def moveSelectedBoard(self):
        detail_url='http://m.cafe.naver.com/ArticleList.nhn?search.clubid={}&search.menuid={}&search.boardtype=L'.format(self.club_id,self.menuid)
        self.driver.get(detail_url)
        time.sleep(3)

    #선택된 게시판 전체글의 아티클아이디를 수집
    def getAllArticlIDSelectedBoard(self):
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        tag_list = soup.find_all('a')
        for tag in tag_list:
            try:
                if re.search("articleid=[0-9]+", tag['href']).group(): #articleid=
                    self.article_id_list.append(re.search("articleid=[0-9]+", tag['href']).group().replace('articleid=',''))
            except:
                continue

    #전체글 검색 모바일에서 페이지별로 아티클 수집, 페이지 최대 1000번째까지
    def getAllArticleIDMobilePages(self):
        for i in range(1, self.pages + 1):
            print("현재 {} 페이지 접속 중".format(i))
            if i!=0 and i%50 ==0:
                time.sleep(20)
            else:
                try:
                    detail_url = 'http://m.cafe.naver.com/{0}/ArticleList.nhn?search.clubid={1}&search.boardtype=L&search.page={2}'.format(self.clubname, self.club_id, i)
                    self.driver.get(detail_url)
                    time.sleep(0.2)
                    html = self.driver.page_source
                    soup = BeautifulSoup(html, 'html.parser')
                    tag_list = soup.find_all('a')
                except:
                    return
                for tag in tag_list:
                    try:
                        if re.search("articleid=[0-9]+", tag['href']).group():
                            self.article_id_list.append(re.search("articleid=[0-9]+", tag['href']).group().replace('articleid=', ''))
                    except:
                        continue

    #모바일버전 전체글(또는 게시판별 전체글) 더보기 시 스크롤 다운
    def pageScrollDown(self):
        for i in range(0, self.more_cnt):
            try:
                self.driver.find_element_by_class_name('u_cbox_page_more').click()
                print("{}번째 더보기 버튼 클릭 중".format(i+1))
                time.sleep(0.5)
            except:
                print('click 할 버튼이 없습니다.')
                continue

    #모바일버전 전체글 보기
    def getAllArticleIdMobile(self):

        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        tag_list = soup.find_all('a')
        for tag in tag_list :
            if re.search("articleid=[0-9]+", tag['href']):
                try:
                    self.article_id_list.append(re.search("articleid=[0-9]+", tag['href']).group().replace('articleid=',''))
                except:
                    continue
            else:
                continue
        print(len(self.article_id_list) , "개 article id를 수집완료")

    #위에서 로그인한 카페에서 키워드 검색
    def searchKeyword(self):
        self.driver.find_element_by_xpath("//a[contains(@onclick,'gnb.cafesearch')]").click()
        time.sleep(5)
        self.driver.find_element_by_id("searchQuery").send_keys(self.keyword)
        self.driver.find_element_by_id("searchQuery").submit()
        time.sleep(3)
        self.userClickSearchBoard()
        self.clickSortOption()

    def clickSortOption(self):
        if self.sort == 2: #정확도를 클릭
            self.driver.find_element_by_id('search_sortBy_2').click()
            time.sleep(1)
            self.driver.find_element_by_xpath("//a[contains(@onclick,'cfs*t.relevant')]").click()
            time.sleep(3)
        else:
            return

    #검색 시 원하는 게시판을 골라서 검색할 수 있다
    def userClickSearchBoard(self):
        if self.choice_board == 1:
            print("원하시는 게시판을 클릭하세요")
            self.driver.find_element_by_id("menuSelectBox").click()
            time.sleep(20)
        else:
            return

    #검색할 때 더보기 버튼 클릭
    def clickMoreButton(self):
        for i in range(self.more_cnt):
            try:
                self.driver.find_element_by_id("moreButton").click()
                print('{}번째 더보기 버튼 클릭 중'.format(i+1))
                time.sleep(0.5)
            except:
                print('click 할 버튼이 없습니다.')
                continue

    #시퀀스 순서유지 중복제거 제너레이터
    def dedupe(self, items):
        seen = list()
        for item in items:
            if item not in seen:
                yield item
                seen.append(item)

    #아티클 댓글 크롤링 하기
    def articleCommentCrawling(self, title, article_id):
        try:
            detail_url = 'http://m.cafe.naver.com/CommentView.nhn?search.clubid={0}&search.articleid={1}&page=1&sc='.format(self.club_id, article_id)
            print("댓글 페이지 진입")
            self.driver.get(detail_url)
            #time.sleep(0.05)
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            tag_list = soup.find_all('span' , class_='u_cbox_contents')

            for tag in tag_list:
                self.data_items[title].append(tag.get_text())
            print("댓글 크롤링 성공")
        except:
            return "댓글 크롤링 실패합니다."

    #아티클 크롤링 하기
    def articleCrwaling(self):
        article_id_backup = copy.deepcopy(self.article_id_list)
        length = len(list(self.dedupe(self.article_id_list)))
        article_id_list = self.dedupe(article_id_backup)
        print("중복 제거된 article id의 갯수 {}".format(length))
        cnt = 1
        for article_id in article_id_list:
                try:
                    if cnt%500==0:
                        time.sleep(20)
                        cnt +=1
                    else:
                        #print("article_id: {}".format(article_id))
                        print("{0}/{1} 크롤링 중".format(cnt,length))
                        detail_url = "http://m.cafe.naver.com/ArticleRead.nhn?clubid={0}&menuid=2&articleid={1}".format(self.club_id, article_id)
                        self.driver.get(detail_url)
                        time.sleep(0.15)
                        html = self.driver.page_source
                        soup = BeautifulSoup(html, 'html.parser')
                        tag_list = soup.find_all('div', id='postContent')  # 네이버 아티클 내용 공통태그
                        re_title_tag = soup.find('h2', class_='tit')  # 일반글 양식의 제목 태그
                        sell_title_tag = soup.find('h4', class_='product_name') # 판매글 양식의 제목 태그
                        # 네이버 일반글 양식의 article 크롤링
                except:
                    return "크롤링을 중단하고 데이터를 저장합니다."

                try:
                    #네이버 일반글 양식의 article 크롤링
                    if re_title_tag:
                        title = re_title_tag.get_text(strip=True).replace('[공식앱]', '') #네이버 양식 제거
                        for tag in tag_list:
                            if self.data_items[title]:
                                print("중복된 게시물 크롤링 시도")
                                continue
                            else:
                                self.data_items[title].append(tag.get_text())
                                if self.isCommentCrwaling == 1:
                                    self.articleCommentCrawling(title, article_id)
                                cnt += 1
                                print("네이버 일반글 크롤링 성공")


                    # 네이버 판매글 양식의 article 크롤링
                    elif sell_title_tag:
                        title = sell_title_tag.get_text().replace('상품명 :', '')
                        #print("정제된 판매글 제목:", title)
                        for tag in tag_list:
                            if self.data_items[title]:
                                print("중복된 게시물 크롤링 시도")
                                continue
                            else:
                                self.data_items[title].append(tag.get_text())
                                if self.isCommentCrwaling == 1:
                                    self.articleCommentCrawling(title, article_id)
                                cnt += 1
                                print("네이버 판매글 크롤링 성공")
                except:
                     self.error_article_id.append(article_id)

        print("크롤링 종료")
        print("==========================================" + "====" * len(self.error_article_id))
        print("[ {0} ] 검색어 크롤링에 실패한 article id : {1}".format(self.keyword, self.error_article_id))
        print("==========================================" + "====" * len(self.error_article_id))

    # 데이터 저장하기
    # self.data_items는 {'글제목' : [내용 댓글]} 과 같은 형태로 되어 있다.
    def data_to_file(self):
        print("txt파일 저장중")
        data = self.data_items
        if self.isCommentCrwaling == 1:
            isComment = '댓글유'
        else:
            isComment = '댓글무'

        if self.keyword != None:
            with open(NaverCafeCrawler.FILE_PATH + "NAVER_CAFE_{0}_{1}_{2}.txt".\
                    format(self.clubname, self.keyword, isComment), "a", encoding="utf-8") as file:

                for key, value in zip(list(data.keys()), list(data.values())):
                    file.write(key + '\n')
                    for v in value:
                        file.write(v + '\n')

                file.close()
        elif self.menuid:
            with open(NaverCafeCrawler.FILE_PATH + "NAVER_CAFE_{0}_{1}_{2}_{3}.txt".format(self.clubname, self.menuid, '전체글', isComment), "a", encoding="utf-8") as file:

                for key, value in zip(list(data.keys()), list(data.values())):
                    file.write(key + '\n')
                    for v in value:
                        file.write(v + '\n')

                file.close()
        else:
            with open(NaverCafeCrawler.FILE_PATH + "NAVER_CAFE_{0}_{1}_{2}.txt". \
                    format(self.clubname, '전체글', isComment), "a",
                      encoding="utf-8") as file:

                for key, value in zip(list(data.keys()), list(data.values())):
                    file.write(key + '\n')
                    for v in value:
                        file.write(v + '\n')

                file.close()

    # 크롤링 수행하기
    def playCrawling(self):

        #키워드 검색
        if self.keyword:
            print("{} 검색중 모바일 버전".format(self.keyword))
            print("총 {}만큼 더보기를 클릭하여 검색합니다.".format(self.more_cnt))
            self.naverLogin()
            self.searchKeyword()
            self.clickMoreButton()
            self.getAllArticleIdMobile()
            self.articleCrwaling()
            self.data_to_file()

        #전체글 검색
        elif self.version == 'mobile' and self.keyword is None:
            print("전체글 검색 모바일 버전")
            print("총 {}만큼 더보기를 클릭하여 검색합니다.".format(self.more_cnt))
            self.naverLogin()
            self.pageScrollDown()
            self.getAllArticleIdMobile()
            self.articleCrwaling()
            self.data_to_file()

        #게시판 별 , 전체글 긁어오기. 더보기버튼
        elif self.version == 'board' and self.keyword is None:
            print("카페 내 특정 게시판의 전체글을 검색합니다.".format(self.more_cnt))
            print("총 {}번 더보기를 클릭하여 검색합니다.".format(self.more_cnt))
            MENU_ID_READER = MenuIdReader()
            self.naverLogin()
            self.menuid = MENU_ID_READER.menuIDPrint(self.clubname)
            self.moveSelectedBoard()
            self.pageScrollDown()
            self.getAllArticlIDSelectedBoard()
            self.articleCrwaling()
            self.data_to_file()

        elif self.version == 'full':
            print("1페이지부터 {}페이지까지 전체글을 검색합니다.".format(self.pages))
            self.naverLogin()
            self.getAllArticleIDMobilePages()
            self.articleCrwaling()
            self.data_to_file()
            self.data_to_file()

    #사용자 메뉴 출력
    def showSearchMenu(self):
        menuboard = "============\n" \
                    "1) 전체글 검색 \n" \
                    "2) 키워드 검색 \n" \
                    "============\n"

        print(menuboard)
        input_num = int(input("원하시는 메뉴를 입력하세요 : "))

        swichmap = {1: '전체글 검색', 2: '키워드 검색'}

        if swichmap[input_num] == '전체글 검색' :
            print(list(NaverCafeCrawler.CLUB_ID_DIC.keys()))
            self.clubname = input('클럽이름을 입력하세요: ')
            self.isCommentCrwaling = int(input('댓글도 크롤링 하시겠습니까? 1.Yes   2.No: '))
            version = int(input(    "============================================\n"
                                 "          [[1. 게시판 선택]] \n"
                                 "          [[2. 전체글 더보기 버튼 단위 검색]] \n"
                                 "          [[3. 전체글 페이지 단위 검색]]\n"
                                 "============================================\n"
                                 "입력: "
                                ))

            if version == 1:
                print('1')
                self.version = 'board'
                self.more_cnt = int(input('더보기 버튼 클릭 횟수:(1회당 20개 게시글): '))

            if version == 2:
                print('2')
                self.version = 'mobile'
                self.more_cnt = int(input('더보기 버튼 클릭 횟수 (1회당 20개 게시글): '))

            elif version == 3:
                print('3')
                self.version = 'full'
                self.pages = int(input('총 페이지 수 입력 (최대1000페이지): '))


        elif swichmap[input_num] == '키워드 검색':

            print(list(NaverCafeCrawler.CLUB_ID_DIC.keys()))
            self.clubname = input('클럽이름을 입력하세요: ')
            self.isCommentCrwaling = int(input('댓글도 크롤링 하시겠습니까? 1.Yes   2.No: '))
            self.keyword = input('키워드를 입력하세요: ')
            self.choice_board = int(input('특정 게시판을 선택하시겠습니까? 1.Yes  2.No: '))
            self.sort = int(input('검색결과를 정렬하시겠습니까? 1.최신순  2.정확도: '))
            self.more_cnt = int(input('더보기 버튼 클릭 횟수? [1회당 20개 게시글]: '))



if __name__ == '__main__':



    n2 = NaverCafeCrawler()

    # 실행 후 맨 밑에서 코드 구동 후 메모리 체크, 시간체크
    end_time = time.time()
    print(round((end_time - start_time)/60,2), "분")
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    after_start = mem[0]
    print('memory use : ', after_start-before_start)
