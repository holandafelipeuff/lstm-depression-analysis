from os.path import dirname, join
import os

ROOT = dirname(dirname(__file__))

PATH_TO_TWITTER_STRATIFIED_DATA = join(ROOT, "data" + os.sep + "twitter_stratified_data.pickle")
PATH_TO_INSTA_STRATIFIED_DATA = join(ROOT, "data" + os.sep + "insta_stratified_data.pickle")


PATH_TO_INSTAGRAM_DATA = join(ROOT, "data" + os.sep + "external" + os.sep + "instagram")
PATH_TO_CLFS_OPTIONS = join(ROOT, "models" + os.sep + "options.json")

PATH_TO_BEST_MODEL = join(ROOT, "models" + os.sep + "best_model")