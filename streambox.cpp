#include <iostream>
#include "avtuner.h"
#include "detector.h"
#include "queue_manager.h"


AVPixelFormat pixfmt_bgr = AV_PIX_FMT_BGR24;
AVPixelFormat pixfmt_yuv = AV_PIX_FMT_YUV420P;
SwsContext* cvt_bgr2yuv = NULL;
SwsContext* cvt_yuv2bgr = NULL;
AVFrame* avframe_src = NULL;
AVFrame* avframe_dst = NULL;
cv::Mat cvframe( FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3 );
QueueManager worker;


bool cvt_init()
{
	bool ret = false;
	int w = FRAME_WIDTH;
	int h = FRAME_HEIGHT;

	cvt_bgr2yuv = sws_getContext( w, h, pixfmt_bgr, w, h, pixfmt_yuv, SWS_BICUBIC, NULL, NULL, NULL );
	cvt_yuv2bgr = sws_getContext( w, h, pixfmt_yuv, w, h, pixfmt_bgr, SWS_BICUBIC, NULL, NULL, NULL );

	if( cvt_bgr2yuv && cvt_yuv2bgr )
	{
		if(( avframe_src = av_frame_alloc()) != NULL && ( avframe_dst = av_frame_alloc()) != NULL )
		{
			if( av_image_alloc( avframe_src->data, avframe_src->linesize, w, h, pixfmt_bgr, 1 ) >= 0 &&
				av_image_alloc( avframe_dst->data, avframe_dst->linesize, w, h, pixfmt_yuv, 1 ) >= 0 )
			{
				avframe_src->width = w;
				avframe_src->height = h;
				avframe_dst->width = w;
				avframe_dst->height = h;
				avframe_dst->format = pixfmt_yuv;
				ret = true;
			}
			else perror("av_image_alloc()");
		}
	}
	return ret;
}

cv::Mat avframe_2_cvmat( AVFrame* frame )
{
	int w = frame->width;
	int h = frame->height;
	AVFrame dst;

	dst.data[0] = (uint8_t*)cvframe.data;
	avpicture_fill( (AVPicture*)&dst, dst.data[0], pixfmt_bgr, w, h );

	sws_scale( cvt_yuv2bgr, frame->data, frame->linesize, 0, h, dst.data, dst.linesize );
	return cvframe;
}

AVFrame* cvmat_2_avframe( cv::Mat* img )
{
	int w = img->cols;
	int h = img->rows;
	
	for( int i = 0; i < h; ++i )
		memcpy( &(avframe_src->data[0][i*avframe_src->linesize[0]]), &((img->data)[i*img->step]), w*3 );
	sws_scale( cvt_bgr2yuv, avframe_src->data, avframe_src->linesize, 0, h, avframe_dst->data, avframe_dst->linesize );
	return avframe_dst;
}

void avframe_update( AVFrame* frame, GPUVARS* g, std::vector<ImgParams>& params )
{
	if( frame && g )
	{
		avframe_2_cvmat( frame );
		img_detect_label( cvframe, params, g );
		cvmat_2_avframe(&cvframe);
	}
}

void infinity_loop()
{
	worker.listen();
}

void avstream_main( AVSTREAMCTX* ctx, char* source, char* destination, char* spec_color )
{
	GPUVARS g;
	g.frame_curr.upload(cvframe);
	std::vector<ImgParams> params = {
		{ 200, 12, cv::Size(3,1), cv::Size(3,3), ALGO_DIFF },
		{ 170, 10, cv::Size(3,1), cv::Size(5,2), ALGO_CURRENT },
		// { 150, 16, cv::Size(3,1), cv::Size(3,3), ALGO_DIFF_GREY },
		// { 120, 12, cv::Size(3,1), cv::Size(2,2), ALGO_CURRENT_GREY },
		// { 10,  12, cv::Size(3,1), cv::Size(3,3), ALGO_DIFF },
		// { 20,  5,  cv::Size(5,1), cv::Size(1,1), ALGO_DIFF },
		// { 170, 10, cv::Size(4,3), cv::Size(8,6), ALGO_CURRENT }
		// { 170, 10, cv::Size(4,3), cv::Size(3,3), ALGO_CURRENT }
	};

	if (spec_color != NULL) {
		if (std::string(spec_color) == "--white") {
			auto new_end = std::remove_if(params.begin(), params.end(), [](ImgParams p) { return (p.algo_t == ALGO_DIFF_GREY || p.algo_t == ALGO_CURRENT_GREY); } );
			if (new_end != params.end()) params.erase(new_end, params.end());
		}
		else if (std::string(spec_color) == "--grey") {
			auto new_end = std::remove_if(params.begin(), params.end(), [](ImgParams p) { return (p.algo_t == ALGO_DIFF || p.algo_t == ALGO_CURRENT); } );
			if (new_end != params.end()) params.erase(new_end, params.end());
		}
	}

	std::thread t1( infinity_loop );
	t1.detach();

	if( ctx != NULL && source != NULL && destination != NULL )
	{
		if( avstream_open_input( source, ctx ) )
		{
			if( avstream_open_output( destination, ctx ) )
			{
				AVPacket pkt, out_pkt;
				while(true)
				{
					av_init_packet(&pkt);
					if( avstream_read_packet( ctx, &pkt ) )
					{
						AVPacket out_pkt;
						av_init_packet(&out_pkt); out_pkt.data = NULL; out_pkt.size = 0;
						if( pkt.stream_index == ctx->v_idx )
						{
							avframe_update( ctx->frame, &g, params );
							// continue;
							if( avstream_encode_video_packet( ctx, &out_pkt, avframe_dst ) )
							{
								if( !avstream_write_packet( ctx, &out_pkt ) )
								{
									av_packet_unref(&out_pkt);
									break;
								}
							}
							else perror( "avstream_encode_video_packet()" );
						}
						else
						{
							// continue;
							if( avstream_encode_audio_packet( ctx, &out_pkt, ctx->frame ) )
							{
								if( !avstream_write_packet2( ctx, &out_pkt ) )
								{
									av_packet_unref(&out_pkt);
									break;
								}
							}
							else perror( "avstream_encode_audio_packet()" );
						}
						av_packet_unref(&out_pkt);
					}
					else
					{
						perror( "avstream_read_packet()" );
						avstream_reopen_input( source, ctx );
					}
					av_packet_unref(&pkt); pkt.data = NULL; pkt.size = 0;
				}
			}
			else perror( "avstream_open_output" );
			avstream_close_output(ctx);
		}
		else
		{
			perror( "avstream_open_input()" );
		}
		avstream_close_input( ctx );
	}
}

int main( int argc, char* argv[] )
{
	int ret = 0;

	if( argc != 3 && argc != 4 ) return 0;

	if( cvt_init() != true ) return 0;
	char* source = argv[1];
	char* destination = argv[2];
	char* spec_color = (argc == 4) ? argv[3] : (char*)"--white_grey";
	AVSTREAMCTX ctx;

	memset( &ctx, 0, sizeof(ctx) );
	tesseract_init();
	avstream_init();

	avstream_main(&ctx, source, destination, spec_color);

	avstream_close_input(&ctx);
	avstream_close_output(&ctx);
}
