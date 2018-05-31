import boto3
import os
import random
import string
import sys
import urllib.request


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
        #self.cos_client = boto3.client('s3', endpoint_url=self.cos_credentials["cos_service_endpoint"])
        self.cos_client = boto3.client('s3', endpoint_url=self.cos_credentials["cos_service_endpoint"], aws_access_key_id=self.cos_credentials["cos_hmac_keys"]["access_key_id"], aws_secret_access_key=self.cos_credentials["cos_hmac_keys"]["secret_access_key"])

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
    def transfer_remote_file_to_bucket(self,file_url,file_name,bucket,save_directory=None):

        # If save directory provided then don't delete local downloads
        working_directory = "temp_cos_files"
        if save_directory is not None:
            working_directory = save_directory

        os.makedirs(working_directory, exist_ok=True)

        # Delete file if present as perhaps download failed and file corrupted
        file_path = os.path.join(working_directory, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

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

    def get_all_objects_in_bucket(self, bucket):

        all_objects = []
        response = self.cos_client.list_objects(Bucket=bucket)
        while response['IsTruncated'] is True:
            # Hit max response limit so get next set of objects
            all_objects = all_objects + response['Contents']
            response = self.cos_client.list_objects(Bucket=bucket, Marker=response['NextMarker'])
        all_objects = all_objects + response['Contents']
        return all_objects

    def download_file(self, bucket, file_to_download, save_path, is_redownload=False):

        if not os.path.exists(save_path) or is_redownload:
            with open(save_path, 'wb') as file:
                print("Downloading %s" % file_to_download)  # "\r" allows us to overwrite the same line
                try:
                    self.cos_client.download_fileobj(bucket, file_to_download, file)
                except:
                    e = sys.exc_info()[0]
                    print(e.__dict__)
                    if e.response != None:
                        print("Detailed error: ", e.response)
                    print('An error occured downloading %s from %s' % (file_to_download, bucket))
                    os.remove(local_file)
                finally:
                    file.close()
