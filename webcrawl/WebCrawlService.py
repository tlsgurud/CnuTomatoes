import math
import re
import requests
from bs4 import BeautifulSoup

def get_movie_title(movie_code):
    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_code)
    result = requests.get(url)
    doc = BeautifulSoup(result.text, 'html.parser')

    title = doc.select('h3.h_movie > a')[0].get_text()
    return title

def calc_pages(movie_code):
    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_code)
    result = requests.get(url)
    doc = BeautifulSoup(result.text, 'html.parser')

    all_count = doc.select('strong.total > em')[0].get_text().strip()
    numbers = re.sub(r'[^0-9]', '', all_count)
    pages = math.ceil(int(all_count) / 10)
    return pages


def get_reviews(movie_code, pages, title):
    count = 0  # Total Review Count

    for page in range(1, pages + 1):
        new_url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code={}&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page={}'.format(movie_code, page)

        result = requests.get(new_url)
        doc = BeautifulSoup(result.text, 'html.parser')

        review_list = doc.select('div.score_result > ul > li')

        for one in review_list:
            count += 1
            print('## USER -> {} #############################################################'.format(count))

            # 평점 정보 수집
            score = one.select('div.star_score > em')[0].get_text()

            # 리뷰 정보 수집
            review = one.select('div.score_reple > p > span')[-1].get_text().strip()  # .strip() #텍스트를 기준으로 좌우 공백제거

            # 작성자(닉네임) 정보 수집
            original_writer = one.select('div.score_reple dt em')[0].get_text().strip()

            idx_end = original_writer.find('(')  # find()# ( 의 인덱스 번호를 줌
            writer = original_writer[0:idx_end]

            # 날짜 정보 수집
            original_date = one.select('div.score_reple dt em')[1].get_text()

            # yyyy.mm.dd 전처리 코드 작성
            date = original_date[:10]
            print('TITLE -> {}'.format(title))
            print('REVIEW -> {}'.format(review))
            print('WRITER -> {}'.format(writer))
            print('SCORE -> {}'.format(score))
            print('DATE -> {}'.format(date))

