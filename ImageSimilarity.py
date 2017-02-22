from sklearn.metrics.pairwise import cosine_similarity

from ImageClassifier import ImageClassifier


class ImageSimilarity:
    def __init__(self):
        self.classifier = ImageClassifier()

    def image_cosine_similarity(self, image_url_1, image_url_2):
        vec_o_1 = self.classifier.run_inference_on_image(image_url_1)[0]
        vec_o_2 = self.classifier.run_inference_on_image(image_url_2)[0]
        return cosine_similarity(vec_o_1, vec_o_2)


    def rank_similar(self, query_image_url, other_image_urls):
        print query_image_url

        sim_results = []

        for other_image_url in other_image_urls:
            similarity = self.image_cosine_similarity(query_image_url, other_image_url)
            sim_results.append([other_image_url, similarity[0][0]])

        sim_results.sort(key=lambda x: x[1], reverse=True)
        for s in sim_results:
            print s


def main():
    url1 = "https://images.viglink.com/product/250x250/www-sneakersnstuff-com/8d8755150e433bde5b89a283acece1d07ca94f2e.jpg?url=http%3A%2F%2Fwww.sneakersnstuff.com%2Fen%2Fimages%2F162480%2Flarge.jpg"
    url2 = "https://images.viglink.com/product/250x250/ecx-images-amazon-com/75246701a29ffe2052c5569cc93e3dedabc772e9.jpg?url=http%3A%2F%2Fecx.images-amazon.com%2Fimages%2FI%2F51Q-n4o1OqL.jpg"
    url3 = "https://images.viglink.com/product/250x250/www-jackrabbit-com/1b81288632460acb335f282d9b1c3a44f67689d2.jpg?url=https%3A%2F%2Fwww.jackrabbit.com%2Fmedia%2Fcatalog%2Fproduct%2Fw%2Fo%2Fwomens-adidas-ultra-boost-3.0-running-shoe-color-core-blackblack-regular-width-size-7-609465303350-01.1891_1.jpg"
    url4 = "https://images.viglink.com/product/250x250/d379fjjlki4dcp-cloudfront-net/fbe790c5a7375938aa54c0131ae4c1a8ee67cc55.jpg?url=https%3A%2F%2Fd379fjjlki4dcp.cloudfront.net%2Fimages%2F207346%2Flarge%2Fadidas-originals-nmdr1-pk-s79168-core-black-the-og.jpg"
    url5 = "https://images.viglink.com/product/250x250/www-sneakersnstuff-com/878ecebc1b47ecdf93293cd23e4491d1c2c9ca39.jpg?url=http%3A%2F%2Fwww.sneakersnstuff.com%2Fen%2Fimages%2F162494%2Flarge.jpg"
    url6 = "https://images.viglink.com/product/250x250/images-na-ssl-images-amazon-com/d05b2d87d0785b2f3060cb6af11ca4f559906c18.jpg?url=https%3A%2F%2Fimages-na.ssl-images-amazon.com%2Fimages%2FI%2F518JXnahWfL.jpg"
    url7 = "https://images.viglink.com/product/250x250/is4-revolveassets-com/8e9351fbd667f6401246f511649419cea06c0408.jpg?url=https%3A%2F%2Fis4.revolveassets.com%2Fimages%2Fp4%2Fn%2Fd%2FASSR-WK5_V1.jpg"
    url8 = "https://images.viglink.com/product/250x250/is4-revolveassets-com/c11d39926afbd20631c1b8b993612f1092cfab98.jpg?url=https%3A%2F%2Fis4.revolveassets.com%2Fimages%2Fp4%2Fn%2Fd%2FLOVF-WO83_V1.jpg"
    url9 = "https://images.viglink.com/product/250x250/is4-revolveassets-com/f42aee55991e413a577e7f2887ddf359bb2bebad.jpg?url=https%3A%2F%2Fis4.revolveassets.com%2Fimages%2Fp4%2Fn%2Fd%2FMCGU-WJ35_V1.jpg"
    url10 = "https://images.viglink.com/product/250x250/crawler-cache-jellolabs-com-imgix-net/746a4e387ccb46fc3f1969db5bd405560c758884.jpg?url=https%3A%2F%2Fcrawler-cache-jellolabs-com.imgix.net%2FPFr4nt5yozZvGLVR%2Fsource_photo.jpg%3Fauto%3Dcompress%252Cformat%26w%3D540%26h%3D800%26fit%3Dclip"
    url11 = "https://images.viglink.com/product/250x250/nord-imgix-net/a2bc8e82303d365aaa6b2b982877012bfd363bf9.jpg?url=http%3A%2F%2Fnord.imgix.net%2FZoom%2F11%2F_100225331.jpg%3Ffit%3Dfill%26bg%3DFFF%26fm%3Djpg%26w%3D380%26h%3D583"
    url12="https://images.viglink.com/product/250x250/content-backcountry-com/8e409332b8d8708d88de5e12fe9d84a70b294be4.jpg?url=http%3A%2F%2Fcontent.backcountry.com%2Fimages%2Fitems%2F900%2FTNF%2FTNF01PD%2FTNMEGRHE.jpg"
    url13="https://images.viglink.com/product/250x250/media-kohlsimg-com/986bd4f8f4cabfa40a3aaccbff5cf325e0063ca9.jpg?url=http%3A%2F%2Fmedia.kohlsimg.com%2Fis%2Fimage%2Fkohls%2F2517922%3Fwid%3D800%26hei%3D800%26op_sharpen%3D1"

    km = ImageSimilarity()
    km.rank_similar(url11, [url1, url2, url3, url4, url5, url6, url7,url8,url9,url10,url11,url12,url13])


if __name__ == "__main__":
    main()