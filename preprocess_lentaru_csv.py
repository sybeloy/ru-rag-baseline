import pandas as pd

if __name__ == '__main__':
    """Укорачиваем датасет (для ускорения итерации тестирования) 
    и оставляем только текст новости
    """
    news_limit = 10000
    data = pd.read_csv('lenta-ru-news.csv')
    data['text'].iloc[:news_limit].to_csv('short.csv')
