import ibm_boto3
import os
import random
import string
import sys
import urllib.request

from ibm_botocore.client import Config


class CosUtils:

    def __init__(self, cos_credentials, region):

        if region is None:
            self.region = "us-south"
        else:
            self.region = region

        self.cos_credentials = cos_credentials
        self.cos_credentials["ibm_auth_endpoint"] = 'https://iam.ng.bluemix.net/oidc/token'
        if "us-south" in self.region:
            self.cos_credentials["cos_service_endpoint"] = 'https://s3-api.us-geo.objectstorage.softlayer.net'
        elif "eu-gb" in self.region:
            self.cos_credentials["cos_service_endpoint"] = 'https://s3.eu-geo.objectstorage.service.networklayer.com'
        else:
            raise ValueError("Region not recognized: %s. Acceptable values are `us-south` or `eu-gb`" % self.region)

        print("cos_service_endpoint: %s" % self.cos_credentials["cos_service_endpoint"])
        self.cos_client = ibm_boto3.client('s3',
                                           ibm_api_key_id=self.cos_credentials["apikey"],
                                           ibm_service_instance_id=self.cos_credentials["resource_instance_id"],
                                           ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
                                           config=Config(signature_version="oauth"),
                                           endpoint_url=self.cos_credentials["cos_service_endpoint"])
    def get_cos_client(self):
        return self.cos_client

    def get_all_buckets(self):
        response = self.cos_client.list_buckets()
        return [bucket['Name'] for bucket in response['Buckets']]

    def get_objects_in_bucket(self,bucket_name):
        return self.cos_client.list_objects(Bucket=bucket_name)

    # Cloud Object Storage (like all object stores) requires that all bucket names be globally unique.  Yes...that's
    # an od quirk but it's the reason object stores are cheap and can scale to terabytes of data.  So we now
    # auto-generate a bucket name that's highly likely to be unique
    def create_unique_bucket(self, bucket_prefix):

        # Create a random 10 digit string
        lst = [random.choice(string.ascii_letters + string.digits) for n in range(10)]
        random_string = "".join(lst).lower()
        bucket = "%s-%s" % (bucket_prefix, random_string)

        self.create_bucket(bucket)
        return bucket

    def create_bucket(self, bucket):

        self.cos_client.create_bucket(Bucket=bucket)
        print('Bucket created: %s' % bucket)

    # Download file from a URL then upload to the given COS bucket
    def transfer_remote_file_to_bucket(self, file_url, file_name, bucket, save_directory=None, redownload=False):

        # If save directory provided then don't delete local downloads
        working_directory = "temp_cos_files"
        if save_directory is not None:
            working_directory = save_directory

        os.makedirs(working_directory, exist_ok=True)

        # Delete file if present as perhaps download failed and file corrupted
        is_downloaded = False
        file_path = os.path.join(working_directory, file_name)
        if os.path.exists(file_path):
            if redownload:
                os.remove(file_path)
            else:
                is_downloaded = True

        if not is_downloaded:
            file_path, _ = urllib.request.urlretrieve(file_url, file_path)
            stat_info = os.stat(file_path)
            print('Downloaded', file_name, stat_info.st_size, 'bytes.')

        print("Uploading %s to bucket: %s" % (file_name,bucket))
        self.cos_client.upload_file(file_path, bucket, file_name)

        # If user provided a save_directory then don't delete the local downloads.
        if save_directory is None:
            if not os.path.exists(file_path):
                os.remove(file_path)
                # Don't delete the download directory itself as calls to this method could be multi-threaded

    def get_all_objects_in_bucket(self, bucket, prefix=None):

        all_objects = []
        response = self.cos_client.list_objects(Bucket=bucket, Prefix=prefix)
        while response['IsTruncated'] is True:
            # Hit max response limit so get next set of objects
            all_objects = all_objects + response['Contents']
            response = self.cos_client.list_objects(Bucket=bucket, Marker=response['NextMarker'])
        all_objects = all_objects + response['Contents']
        return all_objects

    def download_file(self, bucket, file_to_download, save_file, is_redownload=False):

        end_of_path = save_file.rfind(os.sep)
        save_path = save_file[0:end_of_path]
        os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(save_file) or is_redownload:
            with open(save_file, 'wb') as file:
                print("Downloading %s" % file_to_download)  # "\r" allows us to overwrite the same line
                try:
                    self.cos_client.download_fileobj(bucket, file_to_download, file)
                except:
                    e = sys.exc_info()[0]
                    print('An error occured downloading %s from %s' % (file_to_download, bucket))
                    print("Detailed error: ", e)
                    os.remove(local_file)
                finally:
                    file.close()
