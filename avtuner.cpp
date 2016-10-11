#include <stdio.h>
#include <stdlib.h>
#include "avtuner.h"


void log_packet( const AVPacket *pkt )
{
	wprintf( L"pkt->pts = %d\t", pkt->pts );
	wprintf( L"pkt->dts = %d\t", pkt->dts );
	wprintf( L"pkt->duration = %d\t", pkt->duration );
	wprintf( L"pkt->stream_index = %d\n", pkt->stream_index );
}

bool decode_audio_packet( AVSTREAMCTX* ctx, AVPacket* pkt )
{
	bool ret = false;

	if( ctx && pkt )
	{
		int got_frame = 0;
		if( avcodec_decode_audio4( ctx->a_dec, ctx->frame, &got_frame, pkt ) >= 0 )
		{
			// log_packet( pkt );
			if( got_frame )
			{
				ret = true;
			}
			else perror( "got_frame" );
		}
		else perror( "avcodec_decode_audio4()" );
	}
	return ret;
}

bool decode_video_packet( AVSTREAMCTX* ctx, AVPacket* pkt )
{
	bool ret = false;

	if( ctx && pkt )
	{
		int got_frame = 0;
		if( avcodec_decode_video2( ctx->v_dec, ctx->frame, &got_frame, pkt ) >= 0 )
		{
			ret = (bool)got_frame;
		}
		else perror( "avcodec_decode_video2()" );
	}
	return ret;
}

bool avstream_encode_audio_packet( AVSTREAMCTX* ctx, AVPacket* pkt, AVFrame* frame )
{
	bool ret = false;

	if( ctx && pkt && frame )
	{
		int got_packet = 0;
		frame->pts = ctx->a_counter;
		ctx->a_counter += frame->nb_samples;
		if( avcodec_encode_audio2( ctx->a_stream->codec, pkt, frame, &got_packet ) >= 0 )
		{
			pkt->stream_index = ctx->a_idx;
			// log_packet( pkt );
			if( got_packet )
			{
				ret = true;
			}
		}
		else perror("avcodec_encode_audio2()");
	}
	return ret;
}

bool avstream_encode_video_packet( AVSTREAMCTX* ctx, AVPacket* pkt, AVFrame* frame )
{
	bool ret = false;

	if( ctx && pkt && frame )
	{
		int got_packet = 0;
		frame->pts = ctx->v_counter++;
		// wprintf( L"frame->pts = %d\n", frame->pts );
		pkt->data = NULL;
		pkt->size = 0;
		int err = avcodec_encode_video2( ctx->v_stream->codec, pkt, frame, &got_packet );
		if( err >= 0 )
		{
			pkt->stream_index = ctx->v_idx;
			if( got_packet )
			{
				// printf( "Success: got_packet\n" );
				ret = true;
			}
			else perror("got_packet");
		}
		else perror("avcodec_encode_video2()");
		// else printf( "avcodec_encode_video2() = %08X\n", err );
	}
	return ret;
}

void avstream_init()
{
	av_register_all();
	avcodec_register_all();
	avformat_network_init();
}

bool get_stream_decoder( AVFormatContext* in_ctx, AVCodecContext** codec_ctx, int st_idx )
{
	bool ret = false;

	if( in_ctx && codec_ctx && st_idx >= 0 )
	{
		AVStream* in_stream = in_ctx->streams[st_idx];
		AVCodec* dec = NULL;
		if(( dec = avcodec_find_decoder(in_stream->codecpar->codec_id)) )
		{
			if(( *codec_ctx = avcodec_alloc_context3(dec)) )
			{
				if( avcodec_parameters_to_context( *codec_ctx, in_stream->codecpar) >= 0 )
				{
					if( avcodec_open2( *codec_ctx, dec, NULL ) >= 0 )
					{
						ret = true;
					}
					else perror( "avcodec_open2()" );
				}
				else perror( "avcodec_parameters_to_context()" );
			}
			else perror( "avcodec_alloc_context3()" );
		}
		else perror( "avcodec_find_decoder()" );
	}
	return ret;
}

bool get_decoders( AVSTREAMCTX* ctx )
{
	bool ret = 0;

	if( ctx )
	{
		if(( ctx->a_idx = av_find_best_stream( ctx->in, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0 )) >= 0 &&
			( ctx->v_idx = av_find_best_stream( ctx->in, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0 )) >= 0 )
		{
			if( get_stream_decoder( ctx->in, &ctx->a_dec, ctx->a_idx ) && get_stream_decoder( ctx->in, &ctx->v_dec, ctx->v_idx ) )
			{
				ret = true;
			}
			else perror( "get_stream_decoder()" );
		}
		else perror( "av_find_best_stream()" );
	}
	return ret;
}

bool avstream_copy_codec_ctx( AVStream* in_stream, AVStream* out_stream, bool is_global_header_flag )
{
	bool ret = false;

	if( in_stream && out_stream )
	{
		if( avcodec_copy_context( out_stream->codec, in_stream->codec ) >= 0 )
		{
			out_stream->codec->codec_tag = 0;
			if( is_global_header_flag )
				out_stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
			ret = true;
		}
		else perror( "avcodec_copy_context()" );
	}
	return ret;
}

bool avstream_open_input( char* source, AVSTREAMCTX* ctx )
{
	bool ret = false;

	if( source && ctx )
	{
		if( avformat_open_input( &ctx->in, source, NULL, NULL ) >= 0 )
		{
			if( avformat_find_stream_info( ctx->in, NULL ) >= 0 )
			{
				ret = true;
			}
			else perror( "avformat_find_stream_info()" );
		}
		else perror( "avformat_open_input()" );
	}
	return ret;
}

void avstream_reopen_input( char* source, AVSTREAMCTX* ctx )
{
	if( source && ctx )
	{
		avstream_close_input( ctx );
		if( avstream_open_input( source, ctx ) )
		{
			wprintf( L"Reconnected\n" );
		}
		else perror( "avstream_reopen_input()" );
	}
}

bool avstream_open_output( char* dest, PAVSTREAMCTX ctx )
{
	bool ret = false;

	if( dest && ctx )
	{
		AVOutputFormat* out_fmt = NULL;
		avformat_alloc_output_context2( &ctx->out, NULL, "flv", dest );
		if( ctx->out )
		{
			out_fmt = ctx->out->oformat;
			if( get_decoders(ctx) )
			{
				for( int i = 0; i < ctx->in->nb_streams; ++i )
				{
					AVStream* in_stream = ctx->in->streams[i];
					if( i == ctx->v_idx )
					{
						AVCodec* vcodec = avcodec_find_encoder(AV_CODEC_ID_H264);
						// AVCodec* vcodec = avcodec_find_encoder_by_name("h264_nvenc");
						if( vcodec )
						{
							printf( "Video encoder: %s\n", vcodec->name );
							ctx->v_stream = avformat_new_stream( ctx->out, vcodec );
							if( avstream_copy_codec_ctx( in_stream, ctx->v_stream, (out_fmt->flags & AVFMT_GLOBALHEADER) ) == true )
							{
								// if( av_opt_set( ctx->v_stream->codec->priv_data, "preset", "llhq", 0 ) >= 0 )
								if( av_opt_set( ctx->v_stream->codec->priv_data, "preset", "slow", 0 ) >= 0 &&
									av_opt_set( ctx->v_stream->codec->priv_data, "tune", "zerolatency", 0 ) >= 0 )
								{
									ctx->v_stream->codec->width = FRAME_WIDTH;
									ctx->v_stream->codec->height = FRAME_HEIGHT;
									ctx->v_stream->codec->qmin = 10;
									ctx->v_stream->codec->qmax = 51;
									ctx->v_stream->codec->bit_rate = 5*1024*1024;
									// ctx->v_stream->codec->codec_id = AV_CODEC_ID_H264;
									ctx->v_stream->codec->codec_id = vcodec->id;
									ctx->v_stream->codec->pix_fmt = AV_PIX_FMT_YUV420P;
									ctx->v_stream->codec->gop_size = 10;
									ctx->v_stream->codec->max_b_frames = 1;
									ctx->v_stream->codec->time_base = ctx->v_stream->time_base = {1, 25};
									// if( nvenc_encode_init(ctx->v_stream->codec) >=0 )
									{
										if( avcodec_open2( ctx->v_stream->codec, vcodec, NULL ) >= 0 )
										{
										}
										else perror( "avcodec_open2()" );
									}
									// else perror( "nvenc_encode_init()" );
								}
								else perror( "av_opt_set()" );
							}
							else perror( "avstream_copy_codec_ctx()" );
						}
						else perror( "avcodec_find_encoder()" );
					}
					else
					{
						AVCodec* acodec = avcodec_find_encoder(in_stream->codec->codec_id);
						if( acodec )
						{
							ctx->a_stream = avformat_new_stream( ctx->out, acodec );
							if( avstream_copy_codec_ctx( in_stream, ctx->a_stream, (out_fmt->flags & AVFMT_GLOBALHEADER) ) == true )
							{
								// ctx->a_stream->codec->time_base = ctx->a_stream->time_base = {1, 25};
								if( avcodec_open2( ctx->a_stream->codec, acodec, NULL ) >= 0 )
								{
								}
								else perror( "avcodec_open2()" );
							}
							else perror( "avstream_copy_codec_ctx()" );
						}
						else perror( "avcodec_find_encoder()" );
					}
				}
			}
			else perror( "get_decoders()" );
		}
		else perror( "avformat_alloc_output_context2()" );

		if( ctx->a_stream && ctx->v_stream )
		{
			if( !(out_fmt->flags & AVFMT_NOFILE) )
			{
				if( avio_open( &ctx->out->pb, dest, AVIO_FLAG_WRITE ) >= 0 )
				{
					if( avformat_write_header( ctx->out, NULL ) >= 0 )
					{
						ctx->frame = av_frame_alloc();
						if( ctx->frame )
							ret = true;
					}
					else perror( "avformat_write_header()" );
				}
				else perror( "avio_open()" );
			}
		}
		// av_dump_format( ctx->in, 0, "4.flv", 0 );
		// av_dump_format( ctx->out, 0, dest, 1 );
	}
	return ret;
}

void avstream_close_input( AVSTREAMCTX* ctx )
{
	if(ctx->in) { avformat_close_input(&ctx->in); ctx->in = NULL; }
}

void avstream_close_output( AVSTREAMCTX* ctx )
{
	// if(ctx->out)	av_write_trailer(ctx->out);
	if(ctx->a_stream && ctx->a_stream->codec) avcodec_close(ctx->a_stream->codec);
	if(ctx->v_stream && ctx->v_stream->codec) avcodec_close(ctx->v_stream->codec);
	if(ctx->a_dec)	{ avcodec_close(ctx->a_dec); avcodec_free_context(&ctx->a_dec); ctx->a_dec = NULL; }
	if(ctx->v_dec)	{ avcodec_close(ctx->v_dec); avcodec_free_context(&ctx->v_dec); ctx->v_dec = NULL; }
	if(ctx->a_enc)	{ avcodec_close(ctx->a_enc); avcodec_free_context(&ctx->a_enc); ctx->a_enc = NULL; }
	if(ctx->v_enc)	{ avcodec_close(ctx->v_enc); avcodec_free_context(&ctx->v_enc); ctx->v_enc = NULL; }
	if(ctx->frame)	{ av_frame_free(&ctx->frame); ctx->frame = NULL; }
	if(ctx->out)	{ avio_close(ctx->out->pb); avformat_free_context(ctx->out); ctx->out = NULL; }
}

bool avstream_read_packet( AVSTREAMCTX* ctx, AVPacket* pkt )
{
	bool ret = false;

	if( ctx && pkt )
	{
		int err = av_read_frame( ctx->in, pkt );
		if( err >= 0 )
		{
			if( pkt->stream_index == ctx->a_idx )
				ret = decode_audio_packet( ctx, pkt );
			else if( pkt->stream_index == ctx->v_idx )
				ret = decode_video_packet( ctx, pkt );
		}
		else
		{
			wprintf( L"Error: av_read_frame() = %X\n", err );
		}
	}
	return ret;
}

bool avstream_write_packet2( AVSTREAMCTX* ctx, AVPacket* pkt )
{
	bool ret = false;

	if( ctx && pkt )
	{
		AVStream* in_stream = ctx->in->streams[pkt->stream_index];
		AVStream* out_stream = ctx->out->streams[pkt->stream_index];
		
		av_packet_rescale_ts( pkt, out_stream->codec->time_base, out_stream->time_base );
		// log_packet( pkt );

		if( av_interleaved_write_frame( ctx->out, pkt ) >= 0 )
		{
			ret = true;
		}
		else perror("av_interleaved_write_frame()");
	}
	return ret;
}

bool avstream_write_packet( AVSTREAMCTX* ctx, AVPacket* pkt )
{
	bool ret = false;

	if( ctx && pkt )
	{
		AVStream* in_stream = ctx->in->streams[pkt->stream_index];
		AVStream* out_stream = ctx->out->streams[pkt->stream_index];

		// av_packet_rescale_ts( pkt, out_stream->codec->time_base, out_stream->time_base );

		// out_stream->time_base = {1, 25};
/*		pkt->pts = av_rescale_q_rnd( pkt->pts, out_stream->codec->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX) );
		pkt->dts = av_rescale_q_rnd( pkt->dts, out_stream->codec->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX) );
		pkt->duration = av_rescale_q( pkt->duration, out_stream->codec->time_base, out_stream->time_base );
*/
		pkt->pts = av_rescale_q_rnd( pkt->pts, out_stream->codec->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX) );
		pkt->dts = av_rescale_q_rnd( pkt->dts, out_stream->codec->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX) );
		pkt->duration = av_rescale_q( pkt->duration, out_stream->codec->time_base, out_stream->time_base );
		// pkt->pos = -1;

		// log_packet( pkt );
		if( av_interleaved_write_frame( ctx->out, pkt ) >= 0 )
		{
			ret = true;
		}
		else perror("av_interleaved_write_frame()");
	}
	return ret;
}
