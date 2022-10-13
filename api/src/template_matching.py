# ONE_NINE = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
# ZERO_SIX = {'0', '1', '2', '3', '4', '5', '6', '7'}
# KZ_REGIONS = {'Z', 'A', 'C', 'D', 'B', 'E', 'L', 'H', 'M', 'N', 'P', 'R', 'X', 'S', 'T', 'F'}
# YELLOW_REGIONS = {'K', 'M', 'H', 'P', 'F', 'C', 'D', 'T'}
# KG_REGIONS = {'A', 'B', 'F', 'D', 'I', 'N', 'O', 'Z', 'T', 'C', 'S'}
# KG_HPMK = {'H', 'P', 'M', 'K'}
#
#
#
# class TemplateMatching:
#     def __init__(self):
#
#         self.templates = [
#             '###@@@zx',
#             '###@@zx',
#             'r###@@@',
#             'r###@@',
#             '##@@zx',
#             'CMD####',
#             'h######',
#             '###KPzx',
#             '####zx',
#             '###ADM',
#             '###T@@@',
#             '###AV',
#             '##SK',
#             '##KZ',
#             'HC####',
#             '@@#####',
#             '@###@@##',
#             '@###@@###',
#             '@@######',
#             '0n###@@',
#             '0n###@@@',
#             'g####@',
#             'g####@@',
#             '####@@',
#             '####@@@',
#             'KG@@####',
#             '######p',
#             '#####@@@',
#             '##@###@@',
#             'UN###',
#             '##@@###'
#         ]
#
#     def process_square_lp(self, first_text, second_text):
#         for halfTemp1, halfTemp2 in SQUARE_TEMPLATES_HALF_KZ:
#             if self.does_match_template(halfTemp1, first_text) \
#                     and self.does_match_template(halfTemp2, second_text):
#                 return first_text + second_text[2:] + second_text[0:2]
#         return first_text + second_text
#
#     def has_matching_template(self, plate_num, square=False):
#         for template in self.templates:
#             if self.does_match_template(template, plate_num):
#                 if template != '##@@zx' or square:
#                     return True
#         return False
#
#     def does_match_template(self, template, plate_num):
#         plate_num = plate_num.upper()
#         if not plate_num:
#             return False
#
#         if len(template) != len(plate_num):
#             return False
#
#         for i in range(len(template)):
#             t = template[i]
#             c = plate_num[i]
#
#             if not self.is_correct_char_template(t, c, i, plate_num):
#                 return False
#
#         return True
#
#     def is_correct_char_template(self, t, c, ind, plate_num):
#         if t == c:
#             return True
#
#         if t == '#':
#             return c.isdigit()
#
#         if t == '@':
#             return c.isalpha()
#
#         if t == 'z':
#             return c in {'0', '1'}
#
#         if t == 'x':
#             if plate_num[ind - 1] == '0' and c in ONE_NINE:
#                 return True
#             if plate_num[ind - 1] == '1' and c in ZERO_SIX:
#                 return True
#             return False
#
#         if t == 'r':
#             return c in KZ_REGIONS
#
#         if t == 'h':
#             return c in YELLOW_REGIONS
#
#         if t == 'n':
#             return c in ONE_NINE
#
#         if t == 'g':
#             return c in KG_REGIONS
#
#         if t == 'p':
#             return c in KG_HPMK
#
#         return False


COUNTRY_CODES = ['KZ', 'KG', 'UZ', 'RU', 'BY', 'GE', 'AM']
SQUARE_TEMPLATES_HALF_KZ = [("999", "99AAA"),
                            ("999", "99AA"),
                            ("99",  "99AA"),
                            ("A99", "9999")]

class TemplateMatching:
    def __init__(self):
        self.standard_digit = '9'
        self.standard_char = 'A'
        self.templates = [
            ("999AAA99", 'KZ'),
            ("999AA99", 'KZ'),
            ("99AA99", 'KZ'),
            ("999AAA", 'KZ'),
            ("A999AAA", 'KZ'),
            ("A999AA", 'KZ'),
            ("A999AA99", 'RU'),
            ("AA99999", 'RU'),
            ("A999AA999", 'RU'),
            ("AA999A99", 'RU'),
            ("99999AAA", 'KG'),
            ("99999AA", 'KG'),
            ("99A999AA", 'UZ'),
            ("A999999", 'KZ'),
            ("AAA9999", 'KZ'),
            ("AA999", 'KZ'),
            ("999999", 'KZ'),
            ("999999A", 'KG'),
            ("AAAA9999", 'KG'),
            ("99AA999", 'AM'),
            ("9999AA9", 'BY'),
            ("AA999AA", 'GE'),
            # ("999AA",    'KG'),
            ("9999AAA", 'KG'),
            ("A9999AA", 'KG'),
            ("9999AA", 'KZ'),
            # ("99999AAA",'UZ'),
            ("AAA999", 'UZ'),
            # ("AAA9999",'UZ'),
            # ("A999999",'UZ'),
            ("AA9999", 'UZ'),
            ("99A999999", 'UZ'),
            # ("999AA99",'UZ'),
            ("9999AA99", 'UZ'),
            ("A99999999", 'UZ')
        ]

    def standardize_lp(self, platelabel):
        platelabel = list(platelabel)
        for i in range(len(platelabel)):
            if platelabel[i].isdigit() is True:
                platelabel[i] = self.standard_digit
            elif platelabel[i].isalpha() is True:
                platelabel[i] = self.standard_char
        platelabel = ''.join(platelabel)
        return platelabel

    def get_country_code(self, platelabel):
        standard_platelabel = self.standardize_lp(platelabel)
        for template in self.templates:
            if standard_platelabel == template[0]:
                return template[1]
        return "UNIDENTIFIED_COUNTRY"

    def process_square_lp(self, first_text, second_text):
        tophalf = self.standardize_lp(first_text)
        bottomhalf = self.standardize_lp(second_text)
        for halfTemp1, halfTemp2 in SQUARE_TEMPLATES_HALF_KZ:
            if halfTemp1 == tophalf and halfTemp2 == bottomhalf:
                final_lp = first_text + second_text[2:] + second_text[0:2]
            else:
                final_lp = first_text + second_text
        country_code = self.get_country_code(final_lp)
        return final_lp, country_code
