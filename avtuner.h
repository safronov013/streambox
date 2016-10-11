#ifndef __INCLUDED_AVTUNER_H__
#define __INCLUDED_AVTUNER_H__

#pragma comment( lib, "avtuner.lib" )

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

extern "C"
{
	#include <libavcodec/avcodec.h>
	#include <libavformat/avformat.h>
	#include <libavutil/imgutils.h>
	#include <libavutil/pixfmt.h>
	#include <libavutil/opt.h>
	#include <libavutil/timestamp.h>
	#include <libswscale/swscale.h>
	#include <libswresample/swresample.h>
}

#pragma comment( lib, "avcodec.lib" )
#pragma comment( lib, "avformat.lib" )
#pragma comment( lib, "avutil.lib" )
#pragma comment( lib, "swscale.lib" )
#pragma comment( lib, "swresample.lib" )

#pragma comment( lib, "opencv_core2413.lib" )
#pragma comment( lib, "opencv_highgui2413.lib" )

#define FRAME_WIDTH 1920
// #define FRAME_WIDTH 1366
#define FRAME_HEIGHT 1080
// #define FRAME_HEIGHT 768

typedef struct _AVSTREAMCTX_S
{
	AVFormatContext* in;
	AVFormatContext* out;
	AVCodecContext* a_enc;
	AVCodecContext* v_enc;
	AVCodecContext* a_dec;
	AVCodecContext* v_dec;
	AVStream* a_stream;
	AVStream* v_stream;
	AVFrame* frame;
	int a_idx;
	int v_idx;
	int a_counter;
	int v_counter;
} AVSTREAMCTX, *PAVSTREAMCTX;


void avstream_init();
bool avstream_open_input( char* source, PAVSTREAMCTX ctx );
bool avstream_open_output( char* destination, PAVSTREAMCTX ctx );
void avstream_reopen_input( char* source, AVSTREAMCTX* ctx );

void avstream_close( AVSTREAMCTX* ctx );
void avstream_close_input( AVSTREAMCTX* ctx );
void avstream_close_output( AVSTREAMCTX* ctx );
bool avstream_read_packet( PAVSTREAMCTX ctx, AVPacket* pkt );
bool avstream_write_packet( PAVSTREAMCTX ctx, AVPacket* pkt );
bool avstream_write_packet2( PAVSTREAMCTX ctx, AVPacket* pkt );
bool avstream_encode_audio_packet( AVSTREAMCTX* ctx, AVPacket* pkt, AVFrame* frame );
bool avstream_encode_video_packet( AVSTREAMCTX* ctx, AVPacket* pkt, AVFrame* frame );

#endif
