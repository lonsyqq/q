import cv2
import numpy as np
import pytesseract
import time

# Путь к исполняемому файлу Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Загрузка заранее обученных ML-моделей
card_classifier = cv2.ml.SVM_load("card_classifier.xml")
chips_classifier = cv2.ml.SVM_load("chips_classifier.xml")

# Вспомогательные функции

def get_card_string(img):
    """
    Конвертировать изображение карты в строку
    """
    img = cv2.resize(img, (75, 100))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)
    return pytesseract.image_to_string(img, config='--psm 10 -c tessedit_char_whitelist=0123456789AKQJTCDHS')

def get_chip_value(img):
    """
    Конвертировать изображение фишки в значение
    """
    img = cv2.resize(img, (75, 75))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)
    return int(pytesseract.image_to_string(img, config='--psm 10 -c tessedit_char_whitelist=0123456789'))

def get_hand_value(cards):
    """
    Получить значение руки
    """
    suits = [card[1] for card in cards]
    ranks = [card[0] for card in cards]
    flush = len(set(suits)) == 1
    straight = (max(ranks)-min(ranks) == 4) and (len(set(ranks)) == 5)
    straight = straight or (ranks.count(14) == 1 and ranks.count(10) == 1 and ranks.count(11) == 1 and ranks.count(12) == 1 and ranks.count(13) == 1)
    ranks_count = {rank: ranks.count(rank) for rank in set(ranks)}
    four_of_a_kind = 4 in ranks_count.values()
    full_house = sorted(ranks_count.values()) == [2, 3]
    three_of_a_kind = 3 in ranks_count.values()
    pairs = [rank for rank in ranks_count.keys() if ranks_count[rank] == 2]
    two_pair = len(pairs) == 2
    if flush and straight:
        return 8, max(ranks)
    if four_of_a_kind:
        return 7, [rank for rank in ranks_count.keys() if ranks_count[rank] == 4][0], [rank for rank in ranks_count.keys() if ranks_count[rank] == 1][0]
    if full_house:
        return 6, [rank for rank in ranks_count.keys() if ranks_count[rank] == 3][0], [rank for rank in ranks_count.keys() if ranks_count[rank] == 2][0]
    if flush:
        return 5, max(ranks)
    if straight:
        return 4, max(ranks)
    if three_of_a_kind:
        return 3, [rank for rank in ranks_count.keys() if ranks_count[rank] == 3][0], sorted([rank for rank in ranks_count.keys() if ranks_count[rank] == 1], reverse=True)[:2]

