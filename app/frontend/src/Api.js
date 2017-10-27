import axios from 'axios';
import { WORD, SEQ2SEQ } from './config';

export function queryWord(query) {
    return axios
        .get(WORD, {
            params: {
                line: query
            }
        })
        .then(response => {
            return response.data;
        })
}

export function querySeq2Seq(query) {
    return axios
        .get(SEQ2SEQ, {
            params: {
                line: query.split('，').join(',')
            }
        })
        .then(response => {
            return response.data;
        })
}