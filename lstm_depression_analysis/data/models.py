from datetime import datetime, timedelta
# import face_recognition
from lstm_depression_analysis import settings
import os
import json


class Questionnaire:
    """ Model that takes all information from a participant answer to the
    research questionnaire
    """

    def __init__(self, answer_dict):
        """ Every date attribute is according to this pattern: YYY-MM-DD """
        self.form_application_date = answer_dict["form_application_date"]
        self.form_application_date = datetime.strptime(
            self.form_application_date, "%Y-%m-%d"
        ).date()
        self.email = answer_dict["email"]
        self.sex = answer_dict["sex"]
        self.birth_date = answer_dict["birth_date"]
        self.household_income = answer_dict["household_income"]
        self.facebook_hours = answer_dict["facebook_hours"]
        self.twitter_hours = answer_dict["twitter_hours"]
        self.instagram_hours = answer_dict["instagram_hours"]
        self.academic_degree = answer_dict["academic_degree"]
        self.course_name = answer_dict["course_name"]
        self.semesters = answer_dict["semesters"]
        self.scholarship = answer_dict["scholarship"]
        self.accommodation = answer_dict["accommodation"]
        self.works = answer_dict["works"]
        self.depression_diagnosed = answer_dict["depression_diagnosed"]
        self.in_therapy = answer_dict["in_therapy"]
        self.antidepressants = answer_dict["antidepressants"]
        self.twitter_user_name = answer_dict["twitter_user_name"]
        self.instagram_user_name = answer_dict["instagram_user_name"]
        self._bdi = answer_dict["BDI"]
        self.answer_dict = answer_dict

    def get_bdi(self, category=True):
        """ Return the bdi category as a discrete category value if category
        is True, or the BDI original value if category is False.

        Return:
        0 -- for BDI less than 14 (minimal)
        1 -- for BDI greater or equal than 14 and less than 20 (mild)
        2 -- for BDI greater or equal than 20 and less than 29 (moderate)
        3 -- for BDI greater or equal 29 (severe)

        Or the BDI itself if category = False
        """
        if not category:
            return self._bdi
        n = self._bdi
        if n < 14:
            return 0
        elif n >= 14 and n < 20:
            return 1
        elif n >= 20 and n < 29:
            return 2
        else:
            return 3

    def get_binary_bdi(self):
        """ Return the bdi category in a binary spectrum.

        Return:
        0 --- for BDI less then 20
        1 --- for BDI greater or equal than 20
        """
        if self._bdi < 20:
            return 0
        return 1


class Participant:
    """ The generalization of a participant of our study.

    Since every participant answer the questionnaire, has a username and a list
    of posts, this is the most general case for instagram or twitter users.
    """

    def __init__(self, questionnaire, username, posts):
        self.questionnaire = questionnaire
        self.username = username
        self.posts = posts

    def get_answer_dict(self):
        return self.questionnaire.answer_dict

    def get_posts_from_qtnre_answer_date(self, days):
        """ Return all posts made 'days' before the answer date to the
        questionnaire. """
        bdi_answer_date = self.questionnaire.form_application_date
        delta = bdi_answer_date - timedelta(days=days)
        posts_in_range = []
        for post in self.posts:
            if post.date.date() > delta:
                posts_in_range.append(post)
        return posts_in_range


class InstagramUser(Participant):

    """
    Instagram's profile information data model, with the biodemographic
    information
    """

    def __init__(
        self,
        biography,
        followers_count,
        following_count,
        is_private,
        posts_count,
        questionnaire,
        username,
        instagram_posts,
    ):
        """
        Params:
        questionnaire -- It contains the answer to the questionnaire from this
            InstagramUser, which is a Questionnaire object.
        """
        super().__init__(questionnaire, username, instagram_posts)
        self.biography = biography
        self.followers_count = followers_count
        self.following_count = following_count
        self.is_private = is_private
        self.posts_count = posts_count

    def get_answer_dict(self):
        """ Return the questionnaire dict with appended Instagram data.

        Also return the keys of the new appended dict.

        Return:
        qtnre -- New questionnaire dict, with appended data
        cols -- Appended Data keys
        """
        binary_bdi = self.questionnaire.get_binary_bdi()
        instagram_user_data = dict(
            followers_count=self.followers_count,
            following_count=self.following_count,
            posts_count=self.posts_count,
            binary_bdi=binary_bdi,
        )
        qtnre = super().get_answer_dict().copy()
        qtnre.update(instagram_user_data)
        return qtnre, list(instagram_user_data.keys())


class InstagramPost:

    """
    This class models all attributes gathered from the Instagram scraped data.

    Represents an abstraction for the instagram post entity with all its fields

    """

    def __init__(self, img_path_list, caption, likes_count, timestamp, comments_count):
        """
        Params:
        instagram_user -- Object from InstagramUser class.
        img_paths -- A list of image paths related to a post
        """
        self._img_path_list = img_path_list
        self.caption = caption
        self.likes_count = likes_count
        # This date is composed of date and time part.
        self.date = datetime.fromtimestamp(timestamp)
        self.comments_count = comments_count
        self._face_count_list = []
        # self.face_presence = True if face_count > 0 else False

    def get_img_path_list(self):
        return self._img_path_list

    def get_face_count_list(self):
        return self._face_count_list

    def calculate_face_count_list(self):
        """ Calculate face count for each picture in the post.

        Return:
        True -- If the face_count_list is the same size as the img_path_list
        False -- Otherwise
        """
        self._face_count_list = []
        for img_path in self._img_path_list:
            img = self._load_image(img_path)
            face_count = self._get_face_count(img)
            if face_count > 20:
                face_count = 0
            self._face_count_list.append(face_count)

    def _load_image(self, image_path):
        path_to_file = os.path.join(settings.PATH_TO_INSTAGRAM_DATA, image_path)
        return face_recognition.load_image_file(path_to_file)

    def _get_face_count(self, image):
        face_locations = face_recognition.face_locations(image, model="cnn")
        return len(face_locations)

    def get_dict_representation(self):
        fce_cnt_mean = sum(self._face_count_list) // len(self._face_count_list)
        data_dict = dict(
            caption=self.caption,
            likes_count=self.likes_count,
            date=self.date,
            comments_count=self.comments_count,
            face_count=fce_cnt_mean,
        )
        return data_dict

class TwitterUser(Participant):

    """
    Twitter's profile information data model
    """

    def __init__(
        self,
        depression_diagnosed,
        twitter_followers,
        twitter_following,
        twitter_tweets_count,

        twitter_listed_count,
        twitter_account_date_creation,
        twitter_tweets_liked_count,

        questionnaire,
        username,
        twitter_posts,
    ):
        super().__init__(questionnaire, username, twitter_posts)
        self.depression_diagnosed = depression_diagnosed
        self.twitter_followers = twitter_followers
        self.twitter_following = twitter_following
        self.twitter_tweets_count = twitter_tweets_count
        self.twitter_listed_count = twitter_listed_count
        self.twitter_account_date_creation = twitter_account_date_creation
        self.twitter_tweets_liked_count = twitter_tweets_liked_count

class TwitterPost:

    """
    This class models the twitter Post object that may grow in the future
    """

    def __init__(
        self, 
        tweet_text, 
        date, 
        tweet_retweet_count,
        tweet_favorite_count, 
        tweet_hashtags_count, 
        tweet_symbols_count,
        tweet_user_mentions_count, 
        tweet_urls_count
    ):

        self.tweet_text = tweet_text

        self.date = datetime.strptime(date, "%Y-%m-%d")
        
        self.tweet_retweet_count = tweet_retweet_count
        self.tweet_favorite_count = tweet_favorite_count
        self.tweet_hashtags_count = tweet_hashtags_count
        self.tweet_symbols_count = tweet_symbols_count
        self.tweet_user_mentions_count = tweet_user_mentions_count
        self.tweet_urls_count = tweet_urls_count

    def get_tweet_text(self):
        return self.tweet_text
        
    def get_date(self):
        return self.date

    def get_tweet_retweet_count(self):
        return self.tweet_retweet_count

    def get_tweet_favorite_count(self):
        return self.tweet_favorite_count

    def get_tweet_hashtags_count(self):
        return self.tweet_hashtags_count

    def get_tweet_symbols_count(self):
        return self.tweet_symbols_count

    def get_tweet_user_mentions_count(self):
        return self.tweet_user_mentions_count

    def get_tweet_urls_count(self):
        return self.tweet_urls_count

class Config(object):
    def __init__(self, options_path=None):
        if not options_path:
            options_path = settings.PATH_TO_CLFS_OPTIONS

        with open(options_path, "r") as f:
            options_file = json.load(f)
        self.__dict__ = options_file