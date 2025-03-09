% open a .wav file and calculates note start time and frequency in noteinfo
% then changes this info into midi notes that are saved as midi file
% programme deals in microtones
% Tim Pearce 2009
clear global; clear;

% standard values ---------------------------------------------------------
% -------------------------------------------------------------------------
args = argv();
path_mp3code_ac = args{1};
mp3code_ac = args{2};

in_file_end = ".wav";
final_in = strcat(path_mp3code_ac, in_file_end)

storage_folder = "GeneratedMIDI_Pearce/";
out_file_end = ".mid";
final_out = strcat(storage_folder, mp3code_ac, out_file_end)



%----------------------------------------

Fs=44100;           % sample rate

T=1/Fs;             % sample time
wf='sine';          % synth voice - fm, sine, saw, tim - 
amp=0.3;            % amplitude mod
chunksize=8000;     % size of samples considered per FFT
stepsize=500;       % size of step - some overlap
namew=final_in; % mine %'earth';    % name of .wav file
namem=final_out; %'sample.mid';   % name of midi file
tol=10;             % tolerance of individual note pick
cl=chunksize/Fs;    % chunklength (secs)


% open and read .wav file -------------------------------------------------
% -------------------------------------------------------------------------
%y=wavread(namew);
y=audioread(namew); % mine; audioread for octave
% sound(y, Fs);           % plays sound
ns=size(y,1);           % number of samples


% spectrum of whole .wav --------------------------------------------------
%NFFT = 2^nextpow2(ns);              % next power of 2 from length of y
%Y=fft(y(:,1), NFFT)/ns;             % Fast Fourier Transform
%f = Fs/2*linspace(0,1,NFFT/2+1);    % calculate frequency

%subplot(1,1,1)
%plot (f, abs(Y(1:NFFT/2+1 ,1)));    % plot single sided spectrum of whole.wav
%axis([0 20000 0 0.1]);



% split into overlapping chunks of samples and find freq spectrum of each -
% -------------------------------------------------------------------------
% noteinfo = [ frequency(hz), magnitude, starttime(sec), stoptime(sec) ] --
% -------------------------------------------------------------------------
noteinfo = zeros(1,4);

for i = 1:stepsize:ns-chunksize

  % Fast Fourier Transform of chunk
  NFFT = 2^nextpow2(chunksize);            
  Y = fft(y(i:i+chunksize,1), NFFT)/(chunksize);           
  f = Fs/2*linspace(0,1,NFFT/2+1);
  freqmag(:,1) = f; freqmag(:,2) = abs(Y(1:NFFT/2+1));
  
  % work out start and finish time of chunk
  chunkstart = i/Fs;    chunkend = (i + chunksize)/Fs;

  stepend = (i + stepsize)/Fs;              % start of next chunk
  
 % if i==5001
 %   NFFT = 2^nextpow2(chunksize);              % next power of 2 from length of y
 %   Y=fft(y(i:i+chunksize,1), NFFT)/chunksize;             % Fast Fourier Transform
 %   f = Fs/2*linspace(0,1,NFFT/2+1);    % calculate frequency

 %   subplot(1,1,1)
  %  plot (f, abs(Y(1:NFFT/2+1 ,1)));    % plot single sided spectrum of whole.wav
 %   axis([0 20000 0 0.1]);
%  end
  

  if max(freqmag(:,2)>0.02) % only look for notes if there is a frequency spike
    [row, col] = find(freqmag(:,2)>0.02);  % limit of mag detection      %### requires fine tuning ###  0.007 for short? 0.02 for jesus
    limitfreqmag = freqmag(row, :);
     
    sfm = zeros(1,2);                      % ##make sure doesn't block any freq##
    for k=1:size(row,1)
      [row2, col2] = find(limitfreqmag(:,2)==max(limitfreqmag(:,2)));   % find max
      [row5, col5] = find(sfm(:,1)<limitfreqmag(row2,1)+(tol*5) &  sfm(:,1)>limitfreqmag(row2,1)-(tol*5));
      if isempty(row5)==0;                  % if sfm already has a similar freq
        limitfreqmag(row2,:) = [];          % delete
      else                                  % if its far away
        sfm = [sfm; limitfreqmag(row2,:)];  % save
        limitfreqmag(row2,:) = [];          % delete  
      end 
    end  
    sfm=[sfm(2:size(sfm,1),:)];             % ignore 1st row
    
    for p=1:size(sfm,1)                     % now check if all potential saves cropped up before
      [row3, col3] = find(noteinfo(:,1)<sfm(p,1)+tol & noteinfo(:,1)>sfm(p,1)-tol);

      if isempty(row3)==0                       % if had pitch before (so row3 is not empty)
          [row6,col6] = find( noteinfo(row3, 4) == max(noteinfo(row3, 4)) );    % find most recent note of that freq
          if noteinfo(row3(row6),4) < chunkstart        % if old pitch is finished -> new note          
            noteinfo = [noteinfo; sfm(p,:), chunkend-(chunksize/(2*Fs)), chunkend-(chunksize/(2*Fs))];  
          else                                    % if old pitch is not finished -> update
            noteinfo(row3(row6),4) = chunkend-(chunksize/(2*Fs));           % update end time
          end   
      else                                      % if completely new pitch -> new note
        noteinfo = [noteinfo; sfm(p,:), chunkend-(chunksize/(2*Fs)), chunkend-(chunksize/(2*Fs))];
      end
    end    
  end
end

noteinfo=[noteinfo(2:size(noteinfo,1),:)];  % ignore 1st row of zero's

% delete any notes with dur < 0.1 secs
%noteinfo(:,5) = noteinfo(:,4) - noteinfo(:,3);

% I deactivate the note filter, else the files would contain only a handful notes:
%[r,c] = find(noteinfo(:,4) - noteinfo(:,3) < 0.1);
%noteinfo(r,:) = [];


% midi matrix -------------------------------------------------------------
% -------------------------------------------------------------------------
%Mmini = freq2midi2([noteinfo(:,1)]); % calculate midi pitch and bend mag.
f = [noteinfo(:,1)];
m = ((log(f*32/440)/log(2))*12)+9;  % finds closest midi value
dif = round(m) - m;                 % find dif. between closest and act Hz
m = round(m);
decval = round(8192 - (dif * 4096));% this is 14bit dec must bend by
                                    % convert to 2 7bit dec numbers
msb = bitshift(decval, -7);         % MostSigBit
lsb = decval - (msb*(2^7));         % LeastSigBit
Mmini = [m, msb, lsb];

%module_path = 'audio2midi_Pearce_module/Audio2Midi-master/freq2midi2.m';
%Mmini2 = system(["octave ", module_path, " ", "[", mat2str(noteinfo(:,1)), "]"])); % mine
%m5 = size(Mmini2)

N = size(noteinfo,1);                % number of notes
M = zeros(N,6);
M(:,1) = 1;                          % all in track 1

M(:,2) = 1;                          % all in channel 1
M(:,3) = Mmini(:,1);                 % nearest whole midi note
M(:,4) = 90;                         % velocity   % ###this will be noteinfo(:,2)### with scaling fac (max vel is 127?)
M(:,5) = noteinfo(:,3);              % time note on (secs)
M(:,6) = noteinfo(:,4);              % time note off(secs)
M(:,7) = Mmini(:,2);        % extra column-how much bend note MSB 
M(:,8) = Mmini(:,3);        % extra column-how much bend note LSB


% save midi file  ---------------------------------------------------------
% -------------------------------------------------------------------------
%midi_new = matrix2midi2(M);
% matrix2midi2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return a _column_ vector
% (copied from writemidi.m)


function A=encode_int(val,Nbytes)
A = zeros(Nbytes,1);  %ensure col vector (diff from writemidi.m...)
for i=1:Nbytes
  A(i) = bitand(bitshift(val, -8*(Nbytes-i)), 255);
end
endfunction


%if nargin < 2 % margin is the input number of elements for a function
ticks_per_quarter_note = 300;
%end

%if nargin < 3
timesig = [4,2,24,8];
%end

tracks = unique(M(:,1));
Ntracks = length(tracks);

% start building 'midi' struct

if (Ntracks==1)
  midi.format = 0;
else
  midi.format = 1;
end

midi.ticks_per_quarter_note = ticks_per_quarter_note;

tempo = 500000;   % could be set by user, etc...
% (microsec per quarter note)

for i=1:Ntracks
  
  trM = M(i==M(:,1),:);
  
  note_events_onoff = [];
  note_events_n = [];
  note_events_ticktime = [];
 

  % gather all the notes:
  for j=1:size(trM,1)    
    % note bend event:
    note_events_onoff(end+1)    = 2;
    note_events_n(end+1)        = j;
    note_events_ticktime(end+1) = 1e6 * trM(j,5) * ticks_per_quarter_note / tempo;  % ROW 5 WHICH IS TIME OF NOTE BEND ON
    
    % note on event:
    note_events_onoff(end+1)    = 1;
    note_events_n(end+1)        = j;
    note_events_ticktime(end+1) = 1e6 * trM(j,5) * ticks_per_quarter_note / tempo;  % ROW 5 WHICH IS TIME ON
    
    % note off event:
    note_events_onoff(end+1)    = 0;
    note_events_n(end+1)        = j;
    note_events_ticktime(end+1) = 1e6 * trM(j,6) * ticks_per_quarter_note / tempo;  % ROW 6 WHICH IS TIME OFF
  end

  
  msgCtr = 1;
  
  % set tempo...
  midi.track(i).messages(msgCtr).deltatime = 0;
  midi.track(i).messages(msgCtr).type = 81;
  midi.track(i).messages(msgCtr).midimeta = 0;
  midi.track(i).messages(msgCtr).data = encode_int(tempo,3);
  midi.track(i).messages(msgCtr).chan = [];
  msgCtr = msgCtr + 1;

  % set time sig...
  midi.track(i).messages(msgCtr).deltatime = 0;
  midi.track(i).messages(msgCtr).type = 88;
  midi.track(i).messages(msgCtr).midimeta = 0;
  midi.track(i).messages(msgCtr).data = timesig(:);
  midi.track(i).messages(msgCtr).chan = [];
  msgCtr = msgCtr + 1;
  
  % unknown meta event, neccessary for pitch bend range?
 % midi.track(i).messages(msgCtr).deltatime = 0;
 % midi.track(i).messages(msgCtr).type = 33;
 % midi.track(i).messages(msgCtr).midimeta = 0;
 % midi.track(i).messages(msgCtr).data = 1;
 % midi.track(i).messages(msgCtr).chan = [];
 % msgCtr = msgCtr + 1;
  
  
  % set pitch bend range...!!!!!!!!                        PITCH BEND RANGE
  %midi.track(i).messages(msgCtr).deltatime = 0;
  %midi.track(i).messages(msgCtr).type = 176;
  %midi.track(i).messages(msgCtr).midimeta = 1;
  %midi.track(i).messages(msgCtr).data = [ 39 ; 5 ];         % was 100;0
  %midi.track(i).messages(msgCtr).chan = [];
  %msgCtr = msgCtr + 1;
  
  % set pitch bend range...!!!!!!!!                        PITCH BEND RANGE
%  midi.track(i).messages(msgCtr).deltatime = 0;
 % midi.track(i).messages(msgCtr).type = 176;
 %% midi.track(i).messages(msgCtr).midimeta = 1;
 % midi.track(i).messages(msgCtr).data = [ 100 ; 0 ];   
 % midi.track(i).messages(msgCtr).chan = [];
%  msgCtr = msgCtr + 1;  
  
  % set pitch bend range...!!!!!!!!                        PITCH BEND RANGE
 % midi.track(i).messages(msgCtr).deltatime = 0;
 % midi.track(i).messages(msgCtr).type = 176;
 % midi.track(i).messages(msgCtr).midimeta = 1;
 % midi.track(i).messages(msgCtr).data = [ 6 ; 24 ];       % 24 for +/- 12 semitones
 % midi.track(i).messages(msgCtr).chan = [];
 % msgCtr = msgCtr + 1;  
  
  % set pitch bend range...!!!!!!!!                        PITCH BEND RANGE
 % midi.track(i).messages(msgCtr).deltatime = 0;
 % midi.track(i).messages(msgCtr).type = 176;
 %midi.track(i).messages(msgCtr).midimeta = 1;
 % midi.track(i).messages(msgCtr).data = [ 38 ; 0 ];
 % midi.track(i).messages(msgCtr).chan = [];
 % msgCtr = msgCtr + 1;  
  
  
  
  
  [junk,ord] = sort(note_events_ticktime);
  
  prevtick = 0;
  for j=1:length(ord)                       
    
    n = note_events_n(ord(j));
    cumticks = note_events_ticktime(ord(j));
    
    midi.track(i).messages(msgCtr).deltatime = cumticks - prevtick;
    midi.track(i).messages(msgCtr).midimeta = 1; 
    midi.track(i).messages(msgCtr).chan = trM(n,2);
    midi.track(i).messages(msgCtr).used_running_mode = 0;

    if (note_events_onoff(ord(j))==1)
      % note on:
      midi.track(i).messages(msgCtr).type = 144;
      midi.track(i).messages(msgCtr).data = [trM(n,3); trM(n,4)];   % PITCH, VELOCITY
    elseif (note_events_onoff(ord(j))==0)
      %-- note off msg:
      %midi.track(i).messages(msgCtr).type = 128;
      %midi.track(i).messages(msgCtr).data = [trM(n,3); trM(n,4)];
      %-- note on vel=0:
      midi.track(i).messages(msgCtr).type = 128;
      midi.track(i).messages(msgCtr).data = [trM(n,3); 0];
    elseif (note_events_onoff(ord(j))==2)
      %-- note bend on msg:        
      midi.track(i).messages(msgCtr).type = 224;
      midi.track(i).messages(msgCtr).data = [trM(n,8); trM(n,7)];    % LOOK AT COLUMN 7    % LSB, MSB
    end
    msgCtr = msgCtr + 1;
    
    prevtick = cumticks;
  end

  % end of track:
  midi.track(i).messages(msgCtr).deltatime = 0;
  midi.track(i).messages(msgCtr).type = 47;
  midi.track(i).messages(msgCtr).midimeta = 0;
  midi.track(i).messages(msgCtr).data = [];
  midi.track(i).messages(msgCtr).chan = [];
  msgCtr = msgCtr + 1;
  
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% gives midi


% writemidi.m:
% return a _column_ vector

function bytes=encode_var_length(val)

binStr = dec2base(round(val),2);
Nbytes = ceil(length(binStr)/7);

binStr = ['00000000' binStr];
bytes = [];
for i=1:Nbytes
  if (i==1)
    lastbit = '0';
  else
    lastbit = '1';
  end
  B = bin2dec([lastbit binStr(end-i*7+1:end-(i-1)*7)]);
  bytes = [B; bytes];
end
endfunction


function bytes=encode_midi_msg(msg, run_mode)

bytes = [];

if (run_mode ~= 1)
  bytes = msg.type;
  % channel:
  bytes = bytes + msg.chan;  % lower nibble should be chan
end

bytes = [bytes; msg.data];
endfunction

function bytes=encode_meta_msg(msg)

bytes = 255;
bytes = [bytes; msg.type];
bytes = [bytes; encode_var_length(length(msg.data))];
bytes = [bytes; msg.data];
endfunction


filename = namem;
Ntracks = length(midi.track);

for i=1:Ntracks

  databytes_track{i} = [];
  
  for j=1:length(midi.track(i).messages)

    msg = midi.track(i).messages(j);

    msg_bytes = encode_var_length(msg.deltatime);

    if (msg.midimeta==1)

      % check for doing running mode
      run_mode = 0;
      run_mode = msg.used_running_mode;
      
      % should check that prev msg has same type to allow run
      % mode...
      
      
      %      if (j>1 && do_run_mode && msg.type == midi.track(i).messages(j-1).type)
%	run_mode = 1;
%      end


msg_bytes = [msg_bytes; encode_midi_msg(msg, run_mode)];
    
    
    else
      
      msg_bytes = [msg_bytes; encode_meta_msg(msg)];
      
    end

%    disp(msg_bytes')

%if (msg_bytes ~= msg.rawbytes)
%  error('rawbytes mismatch');
%end

    databytes_track{i} = [databytes_track{i}; msg_bytes];
    
  end
end 


% HEADER
% double('MThd') = [77 84 104 100]

function A=encode_int2(val,Nbytes)

for i=1:Nbytes
  A(i) = bitand(bitshift(val, -8*(Nbytes-i)), 255);
end
endfunction

rawbytes = [77 84 104 100 ...
	    0 0 0 6 ...
	    encode_int2(midi.format,2) ...
	    encode_int2(Ntracks,2) ...
	    encode_int2(midi.ticks_per_quarter_note,2) ...
	   ]';

% TRACK_CHUCKS
for i=1:Ntracks
  a = length(databytes_track{i});
  % double('MTrk') = [77 84 114 107]
  tmp = [77 84 114 107 ...
	 encode_int2(length(databytes_track{i}),4) ...
	 databytes_track{i}']';
  rawbytes(end+1:end+length(tmp)) = tmp;
end


% write to file
fid = fopen(filename,'w');
%fwrite(fid,rawbytes,'char');
fwrite(fid,rawbytes,'uint8');
fclose(fid);


%%%%%%%%%%%%%% stores midi

%module_path = "audio2midi_Pearce_module/Audio2Midi-master/matrix2midi2.m";
%midi_new =  system(["octave ", module_path, " ", mat2str(M)])); 
%mm = size(midi_new)
%writemidi(midi_new, namem);   % save calculated midi


fprintf("Generation and storage done \n\n")
